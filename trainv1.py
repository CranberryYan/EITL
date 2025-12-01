# ============================================================
# 直接可用：支持断点续训（恢复 epoch / 模型参数 / optimizer）
# checkpoint 格式：logs/ep010-loss0.091-val_loss0.174.pth
# ============================================================

import os
import math
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.EITLnet import SegFormer, SegFormer_GA
from utils.callbacks import LossHistory, EvalCallback
from utils.train_utils import (
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
    fit_one_epoch,
)
from utils.dataloader import (
    SegmentationDataset_train,
    SegmentationDataset_val,
    seg_dataset_collate,
)


def get_net(num_classes=2, phi='b2', pretrained=True, dual=True,
            use_ga=True, num_img_classes=2):
    if use_ga:
        model = SegFormer_GA(num_classes=num_classes,
                             phi=phi,
                             pretrained=pretrained,
                             dual=dual,
                             num_img_classes=num_img_classes)
    else:
        model = SegFormer(num_classes=num_classes,
                          phi=phi,
                          pretrained=pretrained,
                          dual=dual)
    return model


def set_backbone_trainable(model, trainable: bool = True):
    if not hasattr(model, "backbone"):
        return
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def _strip_module_prefix(state_dict):
    """兼容 DataParallel 保存的 module. 前缀"""
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_resume_checkpoint(model, ckpt_path, device):
    """
    返回: checkpoint(dict), start_epoch(int)
    start_epoch = ckpt_epoch + 1
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容：有人直接 torch.save(model.state_dict())，也有人保存 dict
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    ckpt_epoch = ckpt.get("epoch", -1)
    start_epoch = int(ckpt_epoch) + 1

    print(f"[RESUME] Loaded checkpoint: {ckpt_path}")
    print(f"[RESUME] ckpt_epoch={ckpt_epoch} => start_epoch={start_epoch}")
    if missing:
        print(f"[RESUME] Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[RESUME] Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    return ckpt, start_epoch


if __name__ == "__main__":
    # ----------------- 基本配置 -----------------
    Cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and Cuda else "cpu")

    num_classes = 2
    phi = "b2"
    pretrained = True
    dual = True
    input_shape = [256, 256]

    total_epoch = 100
    batch_size = 12

    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = "adamw"  # ["adam", "adamw", "sgd"]
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = "cos"

    # ----------------- 保存与日志目录 -----------------
    save_period = 5

    # ✅ checkpoint 保存目录（epxxx-xxx.pth 就在这里）
    ckpt_save_dir = "logs"
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # ✅ LossHistory/EvalCallback 日志目录（你原来的）
    exp_root_dir = r"/mnt/e/jwj/1202/log/b2_network/"
    os.makedirs(exp_root_dir, exist_ok=True)

    # ----------------- 是否做验证评估 -----------------
    eval_flag = False
    eval_period = 1

    # ----------------- 数据集路径 -----------------
    data_path = r"/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/"
    train_txt = os.path.join(data_path, "train.txt")
    val_txt = os.path.join(data_path, "val.txt")

    # ----------------- 损失配置 -----------------
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)

    # ----------------- DataLoader 配置 -----------------
    num_workers = 0
    shuffle = True
    local_rank = 0

    # ----------------- 断点续训配置 -----------------
    # ✅ 这里填你的断点权重（为空则从头训练）
    model_path = None

    # ----------------- 构建模型 -----------------
    use_ga = True
    num_img_classes = 2
    model = get_net(num_classes=num_classes,
                    phi=phi,
                    pretrained=pretrained,
                    dual=dual,
                    use_ga=use_ga,
                    num_img_classes=num_img_classes)

    if not pretrained:
        weights_init(model)

    # ----------------- 日志记录器 -----------------
    if local_rank == 0:
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(exp_root_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ----------------- Cuda / DataParallel -----------------
    model_train = model.train()
    if Cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        model_train = torch.nn.DataParallel(model, device_ids=[0])
        model_train = model_train.to(device)

    # ----------------- 读取数据划分 -----------------
    with open(train_txt, "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_txt, "r", encoding="utf-8") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        print("Train images: {}, Val images: {}".format(num_train, num_val))

    # ----------------- LR 线性缩放 -----------------
    nbs = 32
    lr_limit_max = 1e-4 if optimizer_type in ["adam", "adamw"] else 5e-2
    lr_limit_min = 3e-5 if optimizer_type in ["adam", "adamw"] else 5e-4

    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    if local_rank == 0:
        print("Init_lr_fit = {:.3e}, Min_lr_fit = {:.3e}".format(Init_lr_fit, Min_lr_fit))

    # ----------------- optimizer / scheduler -----------------
    optimizer = {
        "adam": optim.Adam(
            model.parameters(),
            Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "adamw": optim.AdamW(
            model.parameters(),
            Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "sgd": optim.SGD(
            model.parameters(),
            Init_lr_fit,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        ),
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, total_epoch)

    # ----------------- Resume：加载模型 + 恢复 optimizer + 设置起始 epoch -----------------
    init_epoch = 0
    checkpoint = None
    if model_path and os.path.exists(model_path):
        checkpoint, init_epoch = load_resume_checkpoint(model, model_path, device)

        # 恢复 optimizer（必须在 optimizer 创建之后）
        if isinstance(checkpoint, dict) and ("optimizer" in checkpoint):
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("[RESUME] Optimizer state restored.")
            except Exception as e:
                print(f"[RESUME][WARN] Optimizer restore failed: {e}")

    # ----------------- Dataset / DataLoader -----------------
    train_dataset = SegmentationDataset_train(
        train_lines, input_shape, num_classes, True, data_path
    )
    val_dataset = SegmentationDataset_val(
        val_lines, input_shape, num_classes, False, data_path
    )

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate,
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate,
    )

    epoch_step = num_train // batch_size
    epoch_step_val = math.ceil(num_val / batch_size)

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小, 无法完成训练, 请检查 batch_size 与数据量设置。")

    # ----------------- EvalCallback -----------------
    if local_rank == 0:
        eval_callback = EvalCallback(
            model,
            input_shape,
            num_classes,
            val_lines,
            data_path,
            log_dir,
            Cuda,
            eval_flag=eval_flag,
            period=eval_period,
        )
    else:
        eval_callback = None

    # ----------------- 冻结/解冻 -----------------
    freeze_epoch = 0

    # ----------------- 训练日志文件 -----------------
    log_path = os.path.join(ckpt_save_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    print("日志将写入:", log_path)

    # ----------------- 训练主循环（从 init_epoch 开始） -----------------
    for epoch in range(init_epoch, total_epoch):
        if epoch < freeze_epoch:
            set_backbone_trainable(model, trainable=False)
            if epoch == init_epoch and local_rank == 0:
                print(f"=> Freeze backbone for first {freeze_epoch} epochs")
        else:
            set_backbone_trainable(model, trainable=True)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer,
            epoch, epoch_step, epoch_step_val,
            gen, gen_val, total_epoch, Cuda,
            dice_loss, focal_loss, cls_weights, num_classes,
            save_period, ckpt_save_dir,   # ✅ checkpoint 都存到 ckpt_save_dir
            local_rank, device,
            lambda_cls=0.4, logger=log_file
        )

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()
        log_file.close()

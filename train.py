import os
import math
import datetime
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
    """
    控制 backbone 是否参与训练, 用于后续“冻结 / 解冻”实验;
    注意：这里接收的是原始 model(不是 DataParallel 包裹后的),
    DataParallel 的 .module 就是这个 model, 所以参数是共享的;
    """
    if not hasattr(model, "backbone"):
        # 兜底：结构里没有 backbone 属性, 就直接返回
        return
    for p in model.backbone.parameters():
        p.requires_grad = trainable

if __name__ == "__main__":
    # ----------------- 基本配置 -----------------
    Cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and Cuda else "cpu")

    num_classes = 2          # 分割的类别数(含背景)
    phi = "b2"               # SegFormer 规模
    pretrained = True        # 是否使用 ImageNet 预训练 backbone
    dual = True              # 是否启用双流结构(RGB Stream + Noise Stream)

    input_shape = [512, 512]

    # 是否从某个 checkpoint 继续训练
    model_path = ""          # 为空表示从头训练(只用 backbone 预训练)

    init_epoch = 0
    total_epoch = 100
    batch_size = 4

    # 基础学习率(针对 nbs=16 的参照 batch size)
    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = "adamw"  # ["adam", "adamw", "sgd"]
    momentum = 0.9
    weight_decay = 1e-2

    # 学习率衰减策略: ["step", "cos"]
    lr_decay_type = "cos"

    # ----------------- 日志与模型保存 -----------------
    save_period = 5
    save_dir = r"/mnt/e/jwj/1202/log/b2_network/"

    # 是否在训练过程中做验证与评估
    eval_flag = False
    eval_period = 1

    # ----------------- 数据集路径 -----------------
    data_path = r"/mnt/e/jwj/my_data/"

    # ----------------- 损失配置 -----------------
    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)

    # ----------------- DataLoader 配置 -----------------
    num_workers = 0          # 为了环境兼容性, 先设 0；Linux 上可以改大一些
    shuffle = True

    # 单机训练, local_rank 固定为 0(保留接口, 方便以后改 DDP)
    local_rank = 0

    # ----------------- 构建与加载模型 -----------------
    use_ga = True          # 打开 GatedAttention + 图像级检测
    num_img_classes = 2    # 二分类：0=正常, 1=篡改/缺陷
    model = get_net(num_classes=num_classes,
                    phi=phi,
                    pretrained=pretrained,
                    dual=dual,
                    use_ga=use_ga,
                    num_img_classes=num_img_classes)

    # 如果不使用 backbone 预训练, 则初始化权重
    if not pretrained:
        weights_init(model)

    # 从 checkpoint 恢复(如果提供了 model_path)
    if model_path != "":
        if local_rank == 0:
            print(f"Load weights from {model_path}.")
        checkpoint = torch.load(model_path, map_location=device)

        model_dict = model.state_dict()
        pretrained_dict = checkpoint["state_dict"]
        init_epoch = checkpoint.get("epoch", init_epoch)

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if (k in model_dict) and (np.shape(model_dict[k]) == np.shape(v)):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print(
                "\nSuccessful Load Key num: {}, sample: {}".format(
                    len(load_key), str(load_key[:10])
                )
            )
            print(
                "Fail To Load Key num: {}, sample: {}\n".format(
                    len(no_load_key), str(no_load_key[:10])
                )
            )

    # ----------------- 创建日志记录器 -----------------
    if local_rank == 0:
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ----------------- Cuda / DataParallel 包装 -----------------
    model_train = model.train()
    if Cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        # 单机多卡可以改成 device_ids=[0,1,...]
        model_train = torch.nn.DataParallel(model, device_ids=[0])
        model_train = model_train.to(device)

    # ----------------- 读取数据划分 -----------------
    train_txt = os.path.join(data_path, "ImageSets/Segmentation/train.txt")
    val_txt = os.path.join(data_path, "ImageSets/Segmentation/val.txt")

    with open(train_txt, "r") as f:
        train_lines = f.readlines()
    with open(val_txt, "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        print("Train images: {}, Val images: {}".format(num_train, num_val))

    # ----------------- 按 batch_size 做线性学习率缩放 -----------------
    # nbs: "nominal batch size" 参考值
    nbs = 32
    lr_limit_max = 1e-4 if optimizer_type in ["adam", "adamw"] else 5e-2
    lr_limit_min = 3e-5 if optimizer_type in ["adam", "adamw"] else 5e-4

    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(
        max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2),
        lr_limit_max * 1e-2,
    )

    if local_rank == 0:
        print("Init_lr_fit = {:.3e}, Min_lr_fit = {:.3e}".format(
            Init_lr_fit, Min_lr_fit
        ))

    # ----------------- 构建优化器与学习率调度 -----------------
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

    # 如果 checkpoint 里保存了优化器状态, 也一并恢复
    if model_path != "" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 学习率 scheduler：返回一个函数 lr_scheduler_func(epoch) -> lr
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type, Init_lr_fit, Min_lr_fit, total_epoch
    )

    # ----------------- 构建 Dataset 与 DataLoader -----------------
    train_dataset = SegmentationDataset_train(
        train_lines, input_shape, num_classes, True, data_path
    )
    val_dataset = SegmentationDataset_val(
        val_lines, input_shape, num_classes, False, data_path
    )

    # 这里没有用分布式, 所以 sampler 设为 None
    train_sampler = None
    val_sampler = None

    # 训练集：shuffle=True, drop_last=True
    gen = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None and shuffle),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate,
        sampler=train_sampler,
    )

    # 验证集：一般不 shuffle, drop_last=False 避免丢样本
    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate,
        sampler=val_sampler,
    )

    # 迭代次数(train 用 // 因为 drop_last=True；val 用 ceil)
    epoch_step = num_train // batch_size
    epoch_step_val = math.ceil(num_val / batch_size)

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError(
            "数据集过小, 无法完成训练, 请检查 batch_size 与数据量设置。"
        )

    # ----------------- 构建评估回调 -----------------
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

    # ----------------- 可选：前若干 epoch 冻结 backbone 的策略 -----------------
    # 目前 freeze_epoch 设为 0 表示不冻结；你之后可以改成 10 之类做对比实验。
    freeze_epoch = 0

    save_dir = "logs"       # 也可以用你已有的 save_dir
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(
        save_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    # buffering=1 行缓冲, 写一行就刷新一次
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    print("日志将写入:", log_path)

    # ----------------- 训练主循环 -----------------
    for epoch in range(init_epoch, total_epoch):
        # 冻结 / 解冻 backbone
        if epoch < freeze_epoch:
            set_backbone_trainable(model, trainable=False)
            if epoch == init_epoch and local_rank == 0:
                print("=> Freeze backbone for first {} epochs".format(freeze_epoch))
        else:
            set_backbone_trainable(model, trainable=True)

        # 根据 epoch 设置当前学习率
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # 进入单个 epoch 的训练与验证
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer,
                      epoch, epoch_step, epoch_step_val,
                      gen, gen_val, total_epoch, Cuda,
                      dice_loss, focal_loss, cls_weights, num_classes,
                      save_period, save_dir,
                      local_rank, device,
                      lambda_cls=0.4, logger=log_file)   # ★ 把 log_file 传进去)

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()
        log_file.close()

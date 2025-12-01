import os
import math
import numpy as np
from PIL import Image
from functools import partial
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # dice loss
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mask_to_gray_rgb(mask_2d: np.ndarray) -> np.ndarray:
    """
    mask_2d: [H,W]，可以是 0/1、0~255 或任意实数
    输出: 3 通道灰度图，按 min-max 归一化，不做二值化
    """
    m = mask_2d.astype(np.float32)
    m_min, m_max = m.min(), m.max()
    if m_max > m_min:
        m = (m - m_min) / (m_max - m_min)
    else:
        m = np.zeros_like(m, dtype=np.float32)

    m = (m * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.stack([m, m, m], axis=-1)   # [H,W,3]
    return rgb


def fit_one_epoch(model_train,
                  model,
                  loss_history,
                  eval_callback,
                  optimizer,
                  epoch,
                  epoch_step,
                  epoch_step_val,
                  gen,
                  gen_val,
                  total_epoch,
                  cuda,
                  dice_loss,
                  focal_loss,
                  cls_weights,
                  num_classes,
                  save_period,
                  save_dir,
                  local_rank=0,
                  device=torch.device('cuda:0'),
                  lambda_cls=1.0,      # 图像级分类 loss 的权重 λ
                  logger=None):        # 日志文件句柄（可选）
    """
    支持两种模型输出：
      1) seg_logits = model_train(image)
      2) seg_logits, cls_logits = model_train(image, return_cls=True)

    ✅ 增强：
      - 图像级分类：TP/TN/FP/FN, Acc, Precision, Recall, F1, AUC（Train/Val）
      - 像素级分割：TP/TN/FP/FN, Acc, Precision, Recall, F1, IoU, AUC(采样估计)（Train/Val）
      - 图像级标签：优先用 dataloader 的 labels（若为 [B] 且 0/1），否则用 mask 推导
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from collections import deque
    from datetime import datetime
    from PIL import Image

    # ---------- 小工具：统一打印 / 写日志 ----------
    def log(msg: str):
        if local_rank != 0:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        if logger is not None:
            logger.write(line + "\n")
            logger.flush()
        else:
            print(line)

    # ---------- 小工具：把 tensor 转成 0~255 的 3 通道图片 ----------
    def tensor_to_img(x: torch.Tensor) -> np.ndarray:
        """
        x: [C,H,W], 任意范围，输出 [H,W,3] uint8
        """
        x = x.detach().cpu().numpy()
        if x.ndim == 3:
            x = np.transpose(x, (1, 2, 0))
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x)
        x = (x * 255.0).clip(0, 255).astype(np.uint8)

        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x

    def mask_to_gray_rgb(mask_2d: np.ndarray) -> np.ndarray:
        """
        把 mask / 概率图转成灰度 3 通道可视化
        - 若输入是 0/1/255 等 uint8：直接映射
        - 若输入是 float 概率图：归一化到 0~255
        输出: [H,W,3] uint8
        """
        m = mask_2d
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()

        m = np.asarray(m)
        if m.ndim == 3:
            m = np.squeeze(m)

        if np.issubdtype(m.dtype, np.floating):
            mn, mx = float(m.min()), float(m.max())
            if mx > mn:
                m = (m - mn) / (mx - mn)
            else:
                m = np.zeros_like(m)
            m_u8 = (m * 255.0).clip(0, 255).astype(np.uint8)
        else:
            if m.max() <= 1:
                m_u8 = (m.astype(np.uint8) * 255)
            else:
                m_u8 = m.clip(0, 255).astype(np.uint8)

        rgb = np.stack([m_u8, m_u8, m_u8], axis=-1)
        return rgb

    # ================== 指标工具（分类/分割共用） ==================
    EPS = 1e-12

    def update_confusion_from_preds(preds: torch.Tensor, gts: torch.Tensor):
        """
        preds/gts: shape 任意（会 flatten），取值 0/1
        """
        preds = preds.view(-1)
        gts = gts.view(-1)
        tp = int(((preds == 1) & (gts == 1)).sum().item())
        tn = int(((preds == 0) & (gts == 0)).sum().item())
        fp = int(((preds == 1) & (gts == 0)).sum().item())
        fn = int(((preds == 0) & (gts == 1)).sum().item())
        return tp, tn, fp, fn

    def compute_prf_from_confusion(tp, tn, fp, fn):
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        prec = tp / max(EPS, (tp + fp))
        rec = tp / max(EPS, (tp + fn))
        f1 = 2 * prec * rec / max(EPS, (prec + rec))
        return acc, prec, rec, f1

    def compute_iou_from_confusion(tp, fp, fn):
        return tp / max(EPS, (tp + fp + fn))

    def auc_roc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        纯 numpy AUC（含 ties 平均秩）
        y_true: 0/1
        y_score: 概率或得分
        """
        y_true = np.asarray(y_true).astype(np.int64)
        y_score = np.asarray(y_score).astype(np.float64)

        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        order = np.argsort(y_score)
        ys = y_score[order]
        yt = y_true[order]

        ranks = np.empty_like(ys, dtype=np.float64)
        i = 0
        r = 1
        while i < len(ys):
            j = i
            while j + 1 < len(ys) and ys[j + 1] == ys[i]:
                j += 1
            avg_rank = (r + (r + (j - i))) / 2.0
            ranks[i:j + 1] = avg_rank
            r += (j - i + 1)
            i = j + 1

        sum_ranks_pos = ranks[yt == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    # ================== 图像级标签生成（优先用 labels，否则用 mask 推） ==================
    def get_img_labels_from_batch(masks: torch.Tensor, labels: torch.Tensor, device_):
        B = masks.size(0)

        if isinstance(labels, torch.Tensor):
            if labels.dim() == 1 and labels.numel() == B:
                if labels.min() >= 0 and labels.max() <= 1:
                    return labels.long().to(device_)
            if labels.dim() == 2 and labels.size(0) == B and labels.size(1) == 1:
                if labels.min() >= 0 and labels.max() <= 1:
                    return labels.view(-1).long().to(device_)

        with torch.no_grad():
            img_labels = (masks.view(B, -1).max(dim=1)[0] > 0).long().to(device_)
        return img_labels

    # ================== 分割：从 seg_outputs/masks 得到像素级 pred/gt（二值） ==================
    def seg_pred_gt_binary(seg_outputs: torch.Tensor, masks: torch.Tensor):
        """
        返回 pred_fg, gt_fg (0/1, bool)
        - pred_fg：预测前景（篡改区域）
        - gt_fg：真值前景
        兼容 masks: [B,H,W] / [B,1,H,W]
        兼容 seg_outputs: [B,C,H,W] (推荐 C=2)
        """
        # gt
        gt = masks
        if gt.dim() == 4:
            gt = gt[:, 0, :, :]
        gt_fg = (gt > 0)

        # pred
        if seg_outputs.dim() != 4:
            raise ValueError(f"seg_outputs must be [B,C,H,W], got {seg_outputs.shape}")

        C = seg_outputs.size(1)
        if C >= 2:
            pred_cls = torch.argmax(seg_outputs, dim=1)  # [B,H,W]
            pred_fg = (pred_cls > 0)
        else:
            # 若 C=1：当作 sigmoid 输出
            prob = torch.sigmoid(seg_outputs[:, 0, :, :])
            pred_fg = (prob > 0.5)

        return pred_fg, gt_fg

    def seg_prob_pos(seg_outputs: torch.Tensor):
        """
        返回前景概率图 prob_pos: [B,H,W] in [0,1]
        用于像素级 AUC（采样估计）
        """
        C = seg_outputs.size(1)
        if C >= 2:
            prob = torch.softmax(seg_outputs, dim=1)[:, 1, :, :]
        else:
            prob = torch.sigmoid(seg_outputs[:, 0, :, :])
        return prob

    def sample_for_auc(prob_map: torch.Tensor, gt_fg: torch.Tensor, k: int):
        """
        prob_map: [B,H,W] float
        gt_fg:   [B,H,W] bool
        采样 k 个像素，返回 (scores_np, gts_np)
        """
        # flatten
        s = prob_map.reshape(-1)
        y = gt_fg.reshape(-1).to(dtype=torch.int64)

        n = s.numel()
        if n == 0:
            return None, None
        kk = min(k, n)

        # 随机采样索引（比 randperm 更快）
        idx = torch.randint(low=0, high=n, size=(kk,), device=s.device)
        s_samp = s[idx].detach().float().cpu().numpy()
        y_samp = y[idx].detach().cpu().numpy()
        return s_samp, y_samp

    # ===== AUC 采样上限（避免像素太多导致内存爆）=====
    SEG_AUC_MAX_SAMPLES = 200_000
    SEG_AUC_SAMPLES_PER_BATCH = 20_000
    CLS_AUC_MAX_SAMPLES = 200_000  # 图像级也做个上限（一般不会到）

    def cap_concat_samples(scores_list, gts_list, max_n):
        if len(scores_list) == 0:
            return None, None
        s = np.concatenate(scores_list, axis=0)
        y = np.concatenate(gts_list, axis=0)
        if s.shape[0] > max_n:
            idx = np.random.choice(s.shape[0], size=max_n, replace=False)
            s = s[idx]
            y = y[idx]
        return y, s

    # 图像级分类损失 (2 类)
    cls_criterion = nn.CrossEntropyLoss()

    # ======== 训练阶段 ========
    total_loss = 0.0
    total_f_score = 0.0
    total_cls_loss = 0.0
    total_cls_correct = 0
    total_cls_samples = 0

    # 图像级混淆矩阵（Train）
    train_tp = train_tn = train_fp = train_fn = 0
    train_scores = []
    train_gts = []

    # 像素级混淆矩阵（Train）
    train_seg_tp = train_seg_tn = train_seg_fp = train_seg_fn = 0
    train_seg_scores = []
    train_seg_gts = []

    window_size = 10
    train_loss_win = deque(maxlen=window_size)
    train_f_win = deque(maxlen=window_size)
    train_cls_loss_win = deque(maxlen=window_size)

    log_interval = 10
    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        image, masks, labels = batch
        weights = torch.from_numpy(cls_weights)

        if cuda:
            image = image.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

        optimizer.zero_grad()

        # forward
        try:
            outputs = model_train(image, return_cls=True)
        except TypeError:
            outputs = model_train(image)

        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            seg_outputs, cls_logits = outputs
        else:
            seg_outputs = outputs
            cls_logits = None

        # ------- 分割损失 -------
        if focal_loss:
            seg_loss = Focal_Loss(seg_outputs, masks, weights, num_classes=num_classes)
        else:
            seg_loss = CE_Loss(seg_outputs, masks, weights, num_classes=num_classes)

        if dice_loss:
            main_dice = Dice_loss(seg_outputs, labels)
            seg_loss = seg_loss + main_dice

        # ------- 图像级分类损失 + 总损失 -------
        if cls_logits is not None:
            img_labels = get_img_labels_from_batch(masks, labels, device)
            cls_loss = cls_criterion(cls_logits, img_labels)
            loss = seg_loss + lambda_cls * cls_loss
        else:
            cls_loss = None
            loss = seg_loss

        # backward + step
        loss.backward()
        optimizer.step()

        # ------- 分割指标：保留你原来的 f_score -------
        with torch.no_grad():
            _f_score = f_score(seg_outputs, masks)

        total_loss += loss.item()
        total_f_score += _f_score.item()

        # ------- 像素级分割：TP/TN/FP/FN + 采样 AUC -------
        with torch.no_grad():
            pred_fg, gt_fg = seg_pred_gt_binary(seg_outputs, masks)
            tp, tn, fp, fn = update_confusion_from_preds(pred_fg.to(torch.int64), gt_fg.to(torch.int64))
            train_seg_tp += tp
            train_seg_tn += tn
            train_seg_fp += fp
            train_seg_fn += fn

            # 采样像素用于 AUC（估计）
            if len(train_seg_scores) == 0 or (sum(x.shape[0] for x in train_seg_scores) < SEG_AUC_MAX_SAMPLES):
                prob_pos = seg_prob_pos(seg_outputs)  # [B,H,W]
                s_samp, y_samp = sample_for_auc(prob_pos, gt_fg, SEG_AUC_SAMPLES_PER_BATCH)
                if s_samp is not None:
                    train_seg_scores.append(s_samp)
                    train_seg_gts.append(y_samp)

        # ------- 图像级分类：TP/TN/FP/FN + 采样 AUC -------
        if cls_logits is not None:
            with torch.no_grad():
                preds = cls_logits.argmax(dim=1)
                correct = (preds == img_labels).sum().item()
                total_cls_correct += correct
                total_cls_samples += img_labels.size(0)
                total_cls_loss += cls_loss.item()

                tp, tn, fp, fn = update_confusion_from_preds(preds, img_labels)
                train_tp += tp
                train_tn += tn
                train_fp += fp
                train_fn += fn

                prob_pos = torch.softmax(cls_logits, dim=1)[:, 1]
                train_scores.append(prob_pos.detach().cpu().numpy())
                train_gts.append(img_labels.detach().cpu().numpy())

        # ------- 更新滑动窗口 -------
        train_loss_win.append(loss.item())
        train_f_win.append(_f_score.item())
        if cls_logits is not None:
            train_cls_loss_win.append(cls_loss.item())

        # ------- 日志 -------
        if local_rank == 0 and ((iteration + 1) % log_interval == 0 or (iteration + 1) == epoch_step):
            current_lr = get_lr(optimizer)

            avg_loss_epoch = total_loss / (iteration + 1)
            avg_f_epoch = total_f_score / (iteration + 1)

            avg_loss_win = sum(train_loss_win) / len(train_loss_win)
            avg_f_win = sum(train_f_win) / len(train_f_win)

            # 像素级分割统计（从 epoch 开始累计到现在）
            seg_acc, seg_p, seg_r, seg_f1 = compute_prf_from_confusion(train_seg_tp, train_seg_tn, train_seg_fp, train_seg_fn)
            seg_iou = compute_iou_from_confusion(train_seg_tp, train_seg_fp, train_seg_fn)

            log_str = (
                f"[Train] Epoch {epoch+1}/{total_epoch} - "
                f"Iter {iteration+1}/{epoch_step} | "
                f"Loss(avg): {avg_loss_epoch:.4f} | F1(avg): {avg_f_epoch:.4f} | "
                f"Loss(win): {avg_loss_win:.4f} | F1(win): {avg_f_win:.4f} | "
                f"SegAcc: {seg_acc:.4f} SegP: {seg_p:.4f} SegR: {seg_r:.4f} SegF1: {seg_f1:.4f} SegIoU: {seg_iou:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if cls_logits is not None:
                cls_acc_epoch = total_cls_correct / max(1, total_cls_samples)
                avg_cls_loss_epoch = total_cls_loss / (iteration + 1)
                avg_cls_loss_win = (sum(train_cls_loss_win) / len(train_cls_loss_win)) if len(train_cls_loss_win) > 0 else avg_cls_loss_epoch

                _, cls_prec, cls_rec, cls_f1 = compute_prf_from_confusion(train_tp, train_tn, train_fp, train_fn)

                log_str += (
                    f" | ClsLoss(cur): {cls_loss.item():.4f} "
                    f"| ClsLoss(avg): {avg_cls_loss_epoch:.4f} "
                    f"| ClsLoss(win): {avg_cls_loss_win:.4f} "
                    f"| ClsAcc(avg): {cls_acc_epoch:.4f} "
                    f"| ClsP: {cls_prec:.4f} ClsR: {cls_rec:.4f} ClsF1: {cls_f1:.4f}"
                )

            log(log_str)

    # ======== 验证阶段 ========
    val_loss = 0.0
    val_f_score = 0.0
    val_cls_loss = 0.0
    val_cls_correct = 0
    val_cls_samples = 0

    # 图像级混淆矩阵（Val）
    val_tp = val_tn = val_fp = val_fn = 0
    val_scores = []
    val_gts = []

    # 像素级混淆矩阵（Val）
    val_seg_tp = val_seg_tn = val_seg_fp = val_seg_fn = 0
    val_seg_scores = []
    val_seg_gts = []

    val_loss_win = deque(maxlen=window_size)
    val_f_win = deque(maxlen=window_size)
    val_cls_loss_win = deque(maxlen=window_size)

    model_train.eval()

    # ---- 每 2 个 epoch 存一次 val 结果 ----
    vis_interval = 2
    if local_rank == 0 and ((epoch + 1) % vis_interval == 0):
        mask_root = os.path.join(save_dir, "mask_result")
        epoch_vis_dir = os.path.join(mask_root, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_vis_dir, exist_ok=True)
    else:
        epoch_vis_dir = None

    max_vis_per_epoch = None
    vis_saved = 0

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        if len(batch) == 4:
            image, masks, labels, img_names = batch
        else:
            image, masks, labels = batch
            B = image.size(0)
            img_names = [
                f"val_e{epoch+1:03d}_it{iteration+1:04d}_i{bi:02d}.png"
                for bi in range(B)
            ]

        weights = torch.from_numpy(cls_weights)

        with torch.no_grad():
            if cuda:
                image = image.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

            try:
                outputs = model_train(image, return_cls=True)
            except TypeError:
                outputs = model_train(image)

            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                seg_outputs, cls_logits = outputs
            else:
                seg_outputs = outputs
                cls_logits = None

            # 分割损失
            if focal_loss:
                seg_loss = Focal_Loss(seg_outputs, masks, weights, num_classes=num_classes)
            else:
                seg_loss = CE_Loss(seg_outputs, masks, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(seg_outputs, labels)
                seg_loss = seg_loss + main_dice

            # 图像级分类损失 + 总损失
            if cls_logits is not None:
                img_labels = get_img_labels_from_batch(masks, labels, device)
                cls_loss = cls_criterion(cls_logits, img_labels)
                loss = seg_loss + lambda_cls * cls_loss

                preds = cls_logits.argmax(dim=1)
                correct = (preds == img_labels).sum().item()
                val_cls_correct += correct
                val_cls_samples += img_labels.size(0)
                val_cls_loss += cls_loss.item()

                tp, tn, fp, fn = update_confusion_from_preds(preds, img_labels)
                val_tp += tp
                val_tn += tn
                val_fp += fp
                val_fn += fn

                prob_pos = torch.softmax(cls_logits, dim=1)[:, 1]
                val_scores.append(prob_pos.detach().cpu().numpy())
                val_gts.append(img_labels.detach().cpu().numpy())
            else:
                cls_loss = None
                loss = seg_loss

            _f_score = f_score(seg_outputs, masks)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            val_loss_win.append(loss.item())
            val_f_win.append(_f_score.item())
            if cls_logits is not None:
                val_cls_loss_win.append(cls_loss.item())

            # ------- 像素级分割：TP/TN/FP/FN + 采样 AUC -------
            pred_fg, gt_fg = seg_pred_gt_binary(seg_outputs, masks)
            tp, tn, fp, fn = update_confusion_from_preds(pred_fg.to(torch.int64), gt_fg.to(torch.int64))
            val_seg_tp += tp
            val_seg_tn += tn
            val_seg_fp += fp
            val_seg_fn += fn

            if len(val_seg_scores) == 0 or (sum(x.shape[0] for x in val_seg_scores) < SEG_AUC_MAX_SAMPLES):
                prob_pos = seg_prob_pos(seg_outputs)
                s_samp, y_samp = sample_for_auc(prob_pos, gt_fg, SEG_AUC_SAMPLES_PER_BATCH)
                if s_samp is not None:
                    val_seg_scores.append(s_samp)
                    val_seg_gts.append(y_samp)

            # ------- 保存三联图 [原图 | GT mask | Pred mask] -------
            if (local_rank == 0 and
                epoch_vis_dir is not None and
                (max_vis_per_epoch is None or vis_saved < max_vis_per_epoch)):

                B = image.size(0)
                for bi in range(B):
                    if max_vis_per_epoch is not None and vis_saved >= max_vis_per_epoch:
                        break

                    img_np = tensor_to_img(image[bi])

                    gt = masks[bi]
                    if gt.ndim == 3:
                        gt = gt.squeeze(0)
                    gt_np = gt.detach().cpu().numpy()
                    gt_rgb = mask_to_gray_rgb(gt_np)

                    pred_logits = seg_outputs[bi]  # [C,H,W]
                    # 前景概率
                    if pred_logits.size(0) >= 2:
                        pred_prob = torch.softmax(pred_logits, dim=0)[1]
                    else:
                        pred_prob = torch.sigmoid(pred_logits[0])
                    pred_np = pred_prob.detach().cpu().numpy()
                    pred_rgb = mask_to_gray_rgb(pred_np)

                    h = min(img_np.shape[0], gt_rgb.shape[0], pred_rgb.shape[0])
                    w = min(img_np.shape[1], gt_rgb.shape[1], pred_rgb.shape[1])
                    img_vis = img_np[:h, :w, :]
                    gt_vis = gt_rgb[:h, :w, :]
                    pred_vis = pred_rgb[:h, :w, :]

                    triplet = np.concatenate([img_vis, gt_vis, pred_vis], axis=1)

                    raw_name = img_names[bi]
                    base_name = os.path.basename(raw_name)
                    out_path = os.path.join(epoch_vis_dir, base_name)

                    Image.fromarray(triplet).save(out_path)
                    vis_saved += 1

        # 验证日志
        if local_rank == 0 and ((iteration + 1) % log_interval == 0 or (iteration + 1) == epoch_step_val):
            current_lr = get_lr(optimizer)
            avg_val_loss_epoch = val_loss / (iteration + 1)
            avg_val_f_epoch = val_f_score / (iteration + 1)

            avg_val_loss_win = sum(val_loss_win) / len(val_loss_win)
            avg_val_f_win = sum(val_f_win) / len(val_f_win)

            seg_acc, seg_p, seg_r, seg_f1 = compute_prf_from_confusion(val_seg_tp, val_seg_tn, val_seg_fp, val_seg_fn)
            seg_iou = compute_iou_from_confusion(val_seg_tp, val_seg_fp, val_seg_fn)

            log_str = (
                f"[Val]   Epoch {epoch+1}/{total_epoch} - "
                f"Iter {iteration+1}/{epoch_step_val} | "
                f"Loss(avg): {avg_val_loss_epoch:.4f} | F1(avg): {avg_val_f_epoch:.4f} | "
                f"Loss(win): {avg_val_loss_win:.4f} | F1(win): {avg_val_f_win:.4f} | "
                f"SegAcc: {seg_acc:.4f} SegP: {seg_p:.4f} SegR: {seg_r:.4f} SegF1: {seg_f1:.4f} SegIoU: {seg_iou:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if cls_logits is not None:
                val_cls_acc_epoch = val_cls_correct / max(1, val_cls_samples)
                avg_val_cls_loss_epoch = val_cls_loss / (iteration + 1)
                avg_val_cls_loss_win = (sum(val_cls_loss_win) / len(val_cls_loss_win)) if len(val_cls_loss_win) > 0 else avg_val_cls_loss_epoch

                _, val_cls_prec, val_cls_rec, val_cls_f1 = compute_prf_from_confusion(val_tp, val_tn, val_fp, val_fn)

                log_str += (
                    f" | ClsLoss(cur): {cls_loss.item():.4f} "
                    f"| ClsLoss(avg): {avg_val_cls_loss_epoch:.4f} "
                    f"| ClsLoss(win): {avg_val_cls_loss_win:.4f} "
                    f"| ClsAcc(avg): {val_cls_acc_epoch:.4f} "
                    f"| ClsP: {val_cls_prec:.4f} ClsR: {val_cls_rec:.4f} ClsF1: {val_cls_f1:.4f}"
                )

            log(log_str)

    # ======== epoch 结束：记录 & 存权重 ========
    if local_rank == 0:
        train_epoch_loss = total_loss / max(1, epoch_step)
        val_epoch_loss = val_loss / max(1, epoch_step_val)

        loss_history.append_loss(epoch + 1, train_epoch_loss, val_epoch_loss)
        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)

        log('Epoch {}/{} | TrainLoss: {:.4f} | ValLoss: {:.4f}'.format(
            epoch + 1, total_epoch, train_epoch_loss, val_epoch_loss))

        # ======== 图像级分类：AUC/PRF 汇总 ========
        y_tr, s_tr = cap_concat_samples(train_scores, train_gts, CLS_AUC_MAX_SAMPLES)
        y_va, s_va = cap_concat_samples(val_scores, val_gts, CLS_AUC_MAX_SAMPLES)

        if y_tr is not None:
            train_auc = auc_roc_binary(y_tr, s_tr)
            train_acc2, train_prec2, train_rec2, train_f12 = compute_prf_from_confusion(train_tp, train_tn, train_fp, train_fn)
        else:
            train_auc = float("nan")
            train_acc2 = train_prec2 = train_rec2 = train_f12 = float("nan")

        if y_va is not None:
            val_auc = auc_roc_binary(y_va, s_va)
            val_acc2, val_prec2, val_rec2, val_f12v = compute_prf_from_confusion(val_tp, val_tn, val_fp, val_fn)
        else:
            val_auc = float("nan")
            val_acc2 = val_prec2 = val_rec2 = val_f12v = float("nan")

        log(
            f"[Cls][Epoch {epoch+1}] "
            f"Train: Acc={train_acc2:.4f} P={train_prec2:.4f} R={train_rec2:.4f} F1={train_f12:.4f} AUC={train_auc:.4f} "
            f"| Conf(tp,tn,fp,fn)=({train_tp},{train_tn},{train_fp},{train_fn}) "
            f"|| Val: Acc={val_acc2:.4f} P={val_prec2:.4f} R={val_rec2:.4f} F1={val_f12v:.4f} AUC={val_auc:.4f} "
            f"| Conf(tp,tn,fp,fn)=({val_tp},{val_tn},{val_fp},{val_fn})"
        )

        # ======== 像素级分割：AUC/PRF/IoU 汇总 ========
        y_tr_s, s_tr_s = cap_concat_samples(train_seg_scores, train_seg_gts, SEG_AUC_MAX_SAMPLES)
        y_va_s, s_va_s = cap_concat_samples(val_seg_scores, val_seg_gts, SEG_AUC_MAX_SAMPLES)

        train_seg_acc, train_seg_p, train_seg_r, train_seg_f1 = compute_prf_from_confusion(
            train_seg_tp, train_seg_tn, train_seg_fp, train_seg_fn
        )
        train_seg_iou = compute_iou_from_confusion(train_seg_tp, train_seg_fp, train_seg_fn)
        train_seg_auc = auc_roc_binary(y_tr_s, s_tr_s) if y_tr_s is not None else float("nan")

        val_seg_acc, val_seg_p, val_seg_r, val_seg_f1 = compute_prf_from_confusion(
            val_seg_tp, val_seg_tn, val_seg_fp, val_seg_fn
        )
        val_seg_iou = compute_iou_from_confusion(val_seg_tp, val_seg_fp, val_seg_fn)
        val_seg_auc = auc_roc_binary(y_va_s, s_va_s) if y_va_s is not None else float("nan")

        log(
            f"[Seg][Epoch {epoch+1}] "
            f"Train: Acc={train_seg_acc:.4f} P={train_seg_p:.4f} R={train_seg_r:.4f} F1={train_seg_f1:.4f} "
            f"IoU={train_seg_iou:.4f} AUC={train_seg_auc:.4f} "
            f"| Conf(tp,tn,fp,fn)=({train_seg_tp},{train_seg_tn},{train_seg_fp},{train_seg_fn}) "
            f"|| Val: Acc={val_seg_acc:.4f} P={val_seg_p:.4f} R={val_seg_r:.4f} F1={val_seg_f1:.4f} "
            f"IoU={val_seg_iou:.4f} AUC={val_seg_auc:.4f} "
            f"| Conf(tp,tn,fp,fn)=({val_seg_tp},{val_seg_tn},{val_seg_fp},{val_seg_fn})"
        )

        # ======== 保存权重 ========
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if ((epoch + 1) % save_period == 0) or ((epoch + 1) == total_epoch):
            torch.save(
                state,
                os.path.join(
                    save_dir,
                    'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_epoch_loss, val_epoch_loss)
                )
            )

        if len(loss_history.val_loss) <= 1 or val_epoch_loss <= min(loss_history.val_loss):
            log('Save best model to best_epoch_weights.pth')
            torch.save(state, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(state, os.path.join(save_dir, "last_epoch_weights.pth"))

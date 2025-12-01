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
    """

    # ---------- 小工具：统一打印 / 写日志 ----------
    def log(msg: str):
        if local_rank != 0:
            return
        # 生成时间戳，例如 2025-12-04 21:37:15
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
        if x.ndim == 3:  # [C,H,W] -> [H,W,C]
            x = np.transpose(x, (1, 2, 0))
        # 简单归一化到 0~1，再映射到 0~255
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x)
        x = (x * 255.0).clip(0, 255).astype(np.uint8)

        # 如果是单通道，转成 3 通道
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        return x

    def mask_to_rgb(mask_2d: np.ndarray) -> np.ndarray:
        """
        mask_2d: [H,W] 0/1 或类别 id
        这里简单处理成 0(黑) / 非0(白) 的灰阶 3 通道
        """
        mask_bin = (mask_2d > 0).astype(np.uint8) * 255
        rgb = np.stack([mask_bin, mask_bin, mask_bin], axis=-1)

    # 图像级分类损失 (2 类)
    cls_criterion = nn.CrossEntropyLoss()

    # ======== 训练阶段 ========
    total_loss = 0.0
    total_f_score = 0.0
    total_cls_loss = 0.0
    total_cls_correct = 0
    total_cls_samples = 0

    # 滑动窗口（最近 n 个 iter）
    window_size = 10
    train_loss_win = deque(maxlen=window_size)
    train_f_win = deque(maxlen=window_size)
    train_cls_loss_win = deque(maxlen=window_size)

    log_interval = 10  # 每 10 个 iter 打一次日志

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # dataloader: (image, masks, labels)
        image, masks, labels = batch
        weights = torch.from_numpy(cls_weights)

        if cuda:
            image = image.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

        optimizer.zero_grad()

        # 优先尝试带 return_cls=True 的接口
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

        # ------- 图像级分类损失 -------
        if cls_logits is not None:
            B = masks.size(0)
            with torch.no_grad():
                # 只要 mask 中有前景，就认为是 tampered(1)，否则 authentic(0)
                img_labels = (masks.view(B, -1).max(dim=1)[0] > 0).long().to(device)

            cls_loss = cls_criterion(cls_logits, img_labels)
            loss = seg_loss + lambda_cls * cls_loss
        else:
            cls_loss = None
            loss = seg_loss

        # ------- 反向 + 更新 -------
        loss.backward()
        optimizer.step()

        # ------- 分割指标 -------
        with torch.no_grad():
            _f_score = f_score(seg_outputs, masks)

        total_loss += loss.item()
        total_f_score += _f_score.item()

        # ------- 分类指标 -------
        if cls_logits is not None:
            with torch.no_grad():
                preds = cls_logits.argmax(dim=1)   # [B]
                correct = (preds == img_labels).sum().item()
                total_cls_correct += correct
                total_cls_samples += img_labels.size(0)
                total_cls_loss += cls_loss.item()

        # ------- 更新滑动窗口 -------
        train_loss_win.append(loss.item())
        train_f_win.append(_f_score.item())
        if cls_logits is not None:
            train_cls_loss_win.append(cls_loss.item())

        # ------- 日志（每 log_interval 次打印一次）-------
        if local_rank == 0 and ((iteration + 1) % log_interval == 0 or (iteration + 1) == epoch_step):
            current_lr = get_lr(optimizer)

            # running average（从 epoch 开始到现在）
            avg_loss_epoch = total_loss / (iteration + 1)
            avg_f_epoch = total_f_score / (iteration + 1)

            # 滑动平均（最近 window_size 个 iter）
            avg_loss_win = sum(train_loss_win) / len(train_loss_win)
            avg_f_win = sum(train_f_win) / len(train_f_win)

            log_str = (
                f"[Train] Epoch {epoch+1}/{total_epoch} - "
                f"Iter {iteration+1}/{epoch_step} | "
                # f"SegLoss(cur): {seg_loss.item():.4f} | "
                # f"TotalLoss(cur): {loss.item():.4f} | "
                # f"F1(cur): {_f_score.item():.4f} | "
                f"Loss(avg): {avg_loss_epoch:.4f} | F1(avg): {avg_f_epoch:.4f} | "
                f"Loss(win): {avg_loss_win:.4f} | F1(win): {avg_f_win:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if cls_logits is not None:
                cls_acc_epoch = total_cls_correct / max(1, total_cls_samples)
                avg_cls_loss_epoch = total_cls_loss / (iteration + 1)
                if len(train_cls_loss_win) > 0:
                    avg_cls_loss_win = sum(train_cls_loss_win) / len(train_cls_loss_win)
                else:
                    avg_cls_loss_win = avg_cls_loss_epoch

                log_str += (
                    f" | ClsLoss(cur): {cls_loss.item():.4f} "
                    f"| ClsLoss(avg): {avg_cls_loss_epoch:.4f} "
                    f"| ClsLoss(win): {avg_cls_loss_win:.4f} "
                    f"| ClsAcc(avg): {cls_acc_epoch:.4f}"
                )

            log(log_str)

    # ======== 验证阶段 ========
    val_loss = 0.0
    val_f_score = 0.0
    val_cls_loss = 0.0
    val_cls_correct = 0
    val_cls_samples = 0

    # 验证用滑动窗口
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

    max_vis_per_epoch = None  # 如果只想存前 N 张，可以改成整数
    vis_saved = 0

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        # 兼容带文件名的 dataloader: (image, masks, labels, names)
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

            # 图像级分类损失
            if cls_logits is not None:
                B = masks.size(0)
                img_labels = (masks.view(B, -1).max(dim=1)[0] > 0).long().to(device)
                cls_loss = cls_criterion(cls_logits, img_labels)
                loss = seg_loss + lambda_cls * cls_loss

                preds = cls_logits.argmax(dim=1)
                correct = (preds == img_labels).sum().item()
                val_cls_correct += correct
                val_cls_samples += img_labels.size(0)
                val_cls_loss += cls_loss.item()
            else:
                cls_loss = None
                loss = seg_loss

            _f_score = f_score(seg_outputs, masks)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            # 更新验证滑动窗口
            val_loss_win.append(loss.item())
            val_f_win.append(_f_score.item())
            if cls_logits is not None:
                val_cls_loss_win.append(cls_loss.item())

            # ------- 保存三联图 [原图 | GT mask | Pred mask] -------
            if (local_rank == 0 and
                epoch_vis_dir is not None and
                (max_vis_per_epoch is None or vis_saved < max_vis_per_epoch)):

                B = image.size(0)
                for bi in range(B):
                    if max_vis_per_epoch is not None and vis_saved >= max_vis_per_epoch:
                        break

                    # 原图
                    img_np = tensor_to_img(image[bi])  # [H,W,3] uint8

                    # GT mask（保持灰度，不二值化）
                    gt = masks[bi]
                    if gt.ndim == 3:
                        gt = gt.squeeze(0)
                    gt_np = gt.detach().cpu().numpy()
                    gt_rgb = mask_to_gray_rgb(gt_np)

                    # Pred mask: 类别 1 的概率图
                    pred_logits = seg_outputs[bi]       # [C,H,W]
                    pred_prob = torch.softmax(pred_logits, dim=0)[1]
                    pred_np = pred_prob.detach().cpu().numpy()
                    pred_rgb = mask_to_gray_rgb(pred_np)

                    h = min(img_np.shape[0], gt_rgb.shape[0], pred_rgb.shape[0])
                    w = min(img_np.shape[1], gt_rgb.shape[1], pred_rgb.shape[1])
                    img_vis  = img_np[:h, :w, :]
                    gt_vis   = gt_rgb[:h, :w, :]
                    pred_vis = pred_rgb[:h, :w, :]

                    triplet = np.concatenate([img_vis, gt_vis, pred_vis], axis=1)

                    raw_name = img_names[bi]
                    base_name = os.path.basename(raw_name)
                    out_path = os.path.join(epoch_vis_dir, base_name)

                    Image.fromarray(triplet).save(out_path)
                    vis_saved += 1

        # 验证日志（每 log_interval 次打印一次）
        if local_rank == 0 and ((iteration + 1) % log_interval == 0 or (iteration + 1) == epoch_step_val):
            current_lr = get_lr(optimizer)
            avg_val_loss_epoch = val_loss / (iteration + 1)
            avg_val_f_epoch = val_f_score / (iteration + 1)

            avg_val_loss_win = sum(val_loss_win) / len(val_loss_win)
            avg_val_f_win = sum(val_f_win) / len(val_f_win)

            log_str = (
                f"[Val]   Epoch {epoch+1}/{total_epoch} - "
                f"Iter {iteration+1}/{epoch_step_val} | "
                # f"Loss(cur): {loss.item():.4f} | "
                # f"F1(cur): {_f_score.item():.4f} | "
                f"Loss(avg): {avg_val_loss_epoch:.4f} | F1(avg): {avg_val_f_epoch:.4f} | "
                f"Loss(win): {avg_val_loss_win:.4f} | F1(win): {avg_val_f_win:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if cls_logits is not None:
                val_cls_acc_epoch = val_cls_correct / max(1, val_cls_samples)
                avg_val_cls_loss_epoch = val_cls_loss / (iteration + 1)
                if len(val_cls_loss_win) > 0:
                    avg_val_cls_loss_win = sum(val_cls_loss_win) / len(val_cls_loss_win)
                else:
                    avg_val_cls_loss_win = avg_val_cls_loss_epoch

                log_str += (
                    f" | ClsLoss(cur): {cls_loss.item():.4f} "
                    f"| ClsLoss(avg): {avg_val_cls_loss_epoch:.4f} "
                    f"| ClsLoss(win): {avg_val_cls_loss_win:.4f} "
                    f"| ClsAcc(avg): {val_cls_acc_epoch:.4f}"
                )

            log(log_str)

    # ======== epoch 结束：记录 & 存权重 ========
    if local_rank == 0:
        train_epoch_loss = total_loss / max(1, epoch_step)
        val_epoch_loss   = val_loss   / max(1, epoch_step_val)

        loss_history.append_loss(epoch + 1, train_epoch_loss, val_epoch_loss)
        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)

        log('Epoch {}/{} | TrainLoss: {:.4f} | ValLoss: {:.4f}'.format(
            epoch + 1, total_epoch, train_epoch_loss, val_epoch_loss))

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # 周期性保存
        if ((epoch + 1) % save_period == 0) or ((epoch + 1) == total_epoch):
            torch.save(
                state,
                os.path.join(
                    save_dir,
                    'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_epoch_loss, val_epoch_loss)
                )
            )

        # best / last
        if len(loss_history.val_loss) <= 1 or val_epoch_loss <= min(loss_history.val_loss):
            log('Save best model to best_epoch_weights.pth')
            torch.save(state, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(state, os.path.join(save_dir, "last_epoch_weights.pth"))

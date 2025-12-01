import os
import math
import numpy as np
from PIL import Image
from functools import partial
from datetime import datetime

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
                  lambda_cls=1.0,
                  logger=None):

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
        return rgb

    # ---------- 图像级分类损失 ----------
    cls_criterion = nn.CrossEntropyLoss()

    # ========== 训练阶段 ==========
    total_loss = 0.0
    total_f_score = 0.0
    total_cls_loss = 0.0
    total_cls_correct = 0
    total_cls_samples = 0

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # 兼容 (image, masks, labels) / (image, masks, labels, names) 两种情况
        if len(batch) == 4:
            image, masks, labels, _ = batch
        else:
            image, masks, labels = batch

        weights = torch.from_numpy(cls_weights)

        if cuda:
            image = image.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

        optimizer.zero_grad()

        # 优先尝试拿到 (seg_logits, cls_logits)
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
                img_labels = (masks.view(B, -1).max(dim=1)[0] > 0).long().to(device)
            cls_loss = cls_criterion(cls_logits, img_labels)
            loss = seg_loss + lambda_cls * cls_loss
        else:
            cls_loss = None
            loss = seg_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _f_score = f_score(seg_outputs, labels)

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if cls_logits is not None:
            with torch.no_grad():
                preds = cls_logits.argmax(dim=1)
                correct = (preds == img_labels).sum().item()
                total_cls_correct += correct
                total_cls_samples += img_labels.size(0)
                total_cls_loss += cls_loss.item()

        if local_rank == 0:
            current_loss = total_loss / (iteration + 1)
            current_f_score = total_f_score / (iteration + 1)
            current_lr = get_lr(optimizer)

            log_str = (f"[Train] Epoch {epoch+1}/{total_epoch} - "
                       f"Iter {iteration+1}/{epoch_step} | "
                       f"SegLoss: {seg_loss.item():.4f} | "
                       f"TotalLoss: {loss.item():.4f} | "
                       f"F-score: {_f_score.item():.4f} | "
                       f"LR: {current_lr:.6f}")

            if cls_logits is not None:
                avg_cls_loss = total_cls_loss / (iteration + 1)
                cls_acc = total_cls_correct / max(1, total_cls_samples)
                log_str += f" | ClsLoss: {cls_loss.item():.4f} (avg {avg_cls_loss:.4f}) | ClsAcc: {cls_acc:.4f}"

            log(log_str)

    # ========== 验证阶段 ==========
    val_loss = 0.0
    val_f_score = 0.0
    val_cls_loss = 0.0
    val_cls_correct = 0
    val_cls_samples = 0

    model_train.eval()

    # ★★★ New: 本 epoch 的可视化目录 & 最多保存多少张 ★★★
    if local_rank == 0:
        mask_root = os.path.join(save_dir, "mask_result")
        epoch_vis_dir = os.path.join(mask_root, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_vis_dir, exist_ok=True)
    else:
        epoch_vis_dir = None

    max_vis_per_epoch = 50  # 如果想保存所有 val 图像，改成 None
    vis_saved = 0

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        # 兼容带文件名的 dataloader：(image, masks, labels, names)
        if len(batch) == 4:
            image, masks, labels, img_names = batch
        else:
            image, masks, labels = batch
            # 如果没有文件名，就自己造一个
            B = image.size(0)
            img_names = [f"val_e{epoch+1:03d}_it{iteration+1:04d}_i{bi:02d}.png"
                         for bi in range(B)]

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
                loss = seg_loss

            _f_score = f_score(seg_outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            # ★★★ New: 保存三联图 [原图 | GT mask | Pred mask] ★★★
            if (local_rank == 0 and
                epoch_vis_dir is not None and
                (max_vis_per_epoch is None or vis_saved < max_vis_per_epoch)):

                B = image.size(0)
                for bi in range(B):
                    if max_vis_per_epoch is not None and vis_saved >= max_vis_per_epoch:
                        break

                    # ---------- 原图 ----------
                    img_np = tensor_to_img(image[bi])  # [H,W,3] uint8

                    # ---------- GT mask ----------
                    gt = masks[bi]
                    if gt.ndim == 3:
                        gt = gt.squeeze(0)            # [H,W]
                    gt_np = gt.detach().cpu().numpy()
                    gt_rgb = mask_to_gray_rgb(gt_np)  # 不二值化，连续灰度

                    # ---------- Pred mask (类别1概率图) ----------
                    pred_logits = seg_outputs[bi]     # [C,H,W]
                    # softmax 按通道做，取 tampered 类(假设为 1)
                    pred_prob = torch.softmax(pred_logits, dim=0)[1]
                    pred_np = pred_prob.detach().cpu().numpy()
                    pred_rgb = mask_to_gray_rgb(pred_np)  # 连续概率 -> 灰度

                    # ---------- 尺寸对齐 ----------
                    h = min(img_np.shape[0], gt_rgb.shape[0], pred_rgb.shape[0])
                    w = min(img_np.shape[1], gt_rgb.shape[1], pred_rgb.shape[1])
                    img_vis  = img_np[:h, :w, :]
                    gt_vis   = gt_rgb[:h, :w, :]
                    pred_vis = pred_rgb[:h, :w, :]

                    triplet = np.concatenate([img_vis, gt_vis, pred_vis], axis=1)

                    # ---------- 文件名 = 原图名 ----------
                    # dataloader 如果给的是路径，就 basename 一下
                    raw_name = img_names[bi]
                    base_name = os.path.basename(raw_name)
                    out_path = os.path.join(epoch_vis_dir, base_name)

                    Image.fromarray(triplet).save(out_path)
                    vis_saved += 1

        # 验证日志
        if local_rank == 0:
            current_val_loss = val_loss / (iteration + 1)
            current_val_f_score = val_f_score / (iteration + 1)
            current_lr = get_lr(optimizer)

            log_str = (f"[Val]   Epoch {epoch+1}/{total_epoch} - "
                       f"Iter {iteration+1}/{epoch_step_val} | "
                       f"Loss: {loss.item():.4f} | "
                       f"F-score: {_f_score.item():.4f} | "
                       f"LR: {current_lr:.6f}")

            if cls_logits is not None:
                avg_val_cls_loss = val_cls_loss / (iteration + 1)
                val_cls_acc = val_cls_correct / max(1, val_cls_samples)
                log_str += f" | ClsLoss: {cls_loss.item():.4f} (avg {avg_val_cls_loss:.4f}) | ClsAcc: {val_cls_acc:.4f}"

            log(log_str)

    # ========== Epoch 结束：记录 & 存权重 ==========
    if local_rank == 0:
        train_epoch_loss = total_loss / epoch_step
        val_epoch_loss = val_loss / epoch_step_val
        loss_history.append_loss(epoch + 1, train_epoch_loss, val_epoch_loss)

        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)

        log('Epoch {}/{} | TrainLoss: {:.4f} | ValLoss: {:.4f}'.format(
            epoch + 1, total_epoch, train_epoch_loss, val_epoch_loss))

        state = {
            "epoch": epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if (epoch + 1) % save_period == 0 or (epoch + 1) == total_epoch:
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

    # ★ NEW: 封装一个 log 函数, 统一写日志(终端 or 文件)
    # def log(msg: str):
    #     if local_rank != 0:
    #         return
    #     if logger is not None:
    #         logger.write(msg + "\n")
    #         logger.flush()
    #     else:
    #         print(msg)

    # # 图像级分类损失(2 类)
    # cls_criterion = nn.CrossEntropyLoss()

    # # --------- 训练阶段 ---------
    # total_loss = 0.0
    # total_f_score = 0.0
    # total_cls_loss = 0.0
    # total_cls_correct = 0
    # total_cls_samples = 0

    # model_train.train()

    # for iteration, batch in enumerate(gen):
    #     if iteration >= epoch_step:
    #         break

    #     # 兼容你的 dataloader：batch = (image, masks, labels)
    #     image, masks, labels = batch
    #     weights = torch.from_numpy(cls_weights)

    #     if cuda:
    #         image = image.to(device)
    #         masks = masks.to(device)
    #         labels = labels.to(device)
    #         weights = weights.to(device)

    #     optimizer.zero_grad()

    #     # ★ 调用模型, 优先尝试拿到 (seg_logits, cls_logits)
    #     try:
    #         outputs = model_train(image, return_cls=True)
    #     except TypeError:
    #         # 模型不支持 return_cls 参数(比如普通 SegFormer)
    #         outputs = model_train(image)

    #     if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
    #         # log('enter picture cls branch')  # 如不需要可以注释掉
    #         seg_outputs, cls_logits = outputs
    #     else:
    #         # log('not enter picture cls branch')
    #         seg_outputs = outputs
    #         cls_logits = None

    #     # -------- 分割损失 --------
    #     if focal_loss:
    #         seg_loss = Focal_Loss(seg_outputs, masks, weights, num_classes=num_classes)
    #     else:
    #         seg_loss = CE_Loss(seg_outputs, masks, weights, num_classes=num_classes)

    #     if dice_loss:
    #         main_dice = Dice_loss(seg_outputs, labels)
    #         seg_loss = seg_loss + main_dice

    #     # -------- 图像级分类损失(如果有 cls_logits) --------
    #     if cls_logits is not None:
    #         B = masks.size(0)
    #         with torch.no_grad():
    #             img_labels = (masks.view(B, -1).max(dim=1)[0] > 0).long().to(device)

    #         # # 如不想在 log 里刷太多细节, 可以直接注释掉下面两行
    #         # for i in img_labels:
    #         #     log(f'img_label: {i}')
    #         #     log(f'cls_logits: {cls_logits}')

    #         cls_loss = cls_criterion(cls_logits, img_labels)
    #         loss = seg_loss + lambda_cls * cls_loss
    #     else:
    #         cls_loss = None
    #         loss = seg_loss

    #     # -------- 反向 & 优化 --------
    #     loss.backward()
    #     optimizer.step()

    #     # -------- 统计分割指标 --------
    #     with torch.no_grad():
    #         _f_score = f_score(seg_outputs, labels)

    #     total_loss += loss.item()
    #     total_f_score += _f_score.item()

    #     # -------- 统计分类指标 --------
    #     if cls_logits is not None:
    #         with torch.no_grad():
    #             preds = cls_logits.argmax(dim=1)   # [B]
    #             # log(f'preds: {preds}')
    #             # log(f'img_labels: {img_labels}')
    #             correct = (preds == img_labels).sum().item()
    #             total_cls_correct += correct
    #             total_cls_samples += img_labels.size(0)
    #             total_cls_loss += cls_loss.item()

    #     # -------- 打印训练日志(改为 log) --------
    #     if local_rank == 0:
    #         current_loss = total_loss / (iteration + 1)
    #         current_f_score = total_f_score / (iteration + 1)
    #         current_lr = get_lr(optimizer)

    #         log_str = (f"[Train] Epoch {epoch+1}/{total_epoch} - "
    #                    f"Iter {iteration+1}/{epoch_step} | "
    #                    f"SegLoss: {seg_loss.item():.4f} | "
    #                    f"TotalLoss: {loss.item():.4f} | "
    #                    f"F-score: {_f_score.item():.4f} | "
    #                    f"LR: {current_lr:.6f}")

    #         if cls_logits is not None:
    #             avg_cls_loss = total_cls_loss / (iteration + 1)
    #             cls_acc = total_cls_correct / max(1, total_cls_samples)
    #             log_str += f" | ClsLoss: {cls_loss.item():.4f} (avg {avg_cls_loss:.4f}) | ClsAcc: {cls_acc:.4f}"

    #         log(log_str)

    # # --------- 验证阶段 ---------
    # val_loss = 0.0
    # val_f_score = 0.0
    # val_cls_loss = 0.0
    # val_cls_correct = 0
    # val_cls_samples = 0

    # model_train.eval()

    # for iteration, batch in enumerate(gen_val):
    #     if iteration >= epoch_step_val:
    #         break

    #     image, masks, labels = batch
    #     weights = torch.from_numpy(cls_weights)

    #     with torch.no_grad():
    #         if cuda:
    #             image = image.to(device)
    #             masks = masks.to(device)
    #             labels = labels.to(device)
    #             weights = weights.to(device)

    #         # 同样尝试拿到 (seg_logits, cls_logits)
    #         try:
    #             outputs = model_train(image, return_cls=True)
    #         except TypeError:
    #             outputs = model_train(image)

    #         if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
    #             seg_outputs, cls_logits = outputs
    #         else:
    #             seg_outputs = outputs
    #             cls_logits = None

    #         # 分割损失
    #         if focal_loss:
    #             seg_loss = Focal_Loss(seg_outputs, masks, weights, num_classes=num_classes)
    #         else:
    #             seg_loss = CE_Loss(seg_outputs, masks, weights, num_classes=num_classes)

    #         if dice_loss:
    #             main_dice = Dice_loss(seg_outputs, labels)
    #             seg_loss = seg_loss + main_dice

    #         # 图像级分类损失
    #         if cls_logits is not None:
    #             B = masks.size(0)
    #             img_labels = (masks.view(B, -1).max(dim=1)[0] > 0).long().to(device)
    #             cls_loss = cls_criterion(cls_logits, img_labels)
    #             loss = seg_loss + lambda_cls * cls_loss

    #             preds = cls_logits.argmax(dim=1)
    #             correct = (preds == img_labels).sum().item()
    #             val_cls_correct += correct
    #             val_cls_samples += img_labels.size(0)
    #             val_cls_loss += cls_loss.item()
    #         else:
    #             loss = seg_loss

    #         _f_score = f_score(seg_outputs, labels)

    #         val_loss += loss.item()
    #         val_f_score += _f_score.item()

    #     # 验证日志
    #     if local_rank == 0:
    #         current_val_loss = val_loss / (iteration + 1)
    #         current_val_f_score = val_f_score / (iteration + 1)
    #         current_lr = get_lr(optimizer)

    #         log_str = (f"[Val]   Epoch {epoch+1}/{total_epoch} - "
    #                    f"Iter {iteration+1}/{epoch_step_val} | "
    #                    f"Loss: {loss.item():.4f} | "
    #                    f"F-score: {_f_score.item():.4f} | "
    #                    f"LR: {current_lr:.6f}")

    #         if cls_logits is not None:
    #             avg_val_cls_loss = val_cls_loss / (iteration + 1)
    #             val_cls_acc = val_cls_correct / max(1, val_cls_samples)
    #             log_str += f" | ClsLoss: {cls_loss.item():.4f} (avg {avg_val_cls_loss:.4f}) | ClsAcc: {val_cls_acc:.4f}"

    #         log(log_str)

    # # --------- 每个 epoch 结束：记录 & 存权重 ---------
    # if local_rank == 0:
    #     train_epoch_loss = total_loss / epoch_step
    #     val_epoch_loss = val_loss / epoch_step_val
    #     loss_history.append_loss(epoch + 1, train_epoch_loss, val_epoch_loss)

    #     if eval_callback is not None:
    #         eval_callback.on_epoch_end(epoch + 1, model_train)

    #     log('Epoch {}/{} | TrainLoss: {:.4f} | ValLoss: {:.4f}'.format(
    #         epoch + 1, total_epoch, train_epoch_loss, val_epoch_loss))

    #     state = {
    #         "epoch": epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }

    #     # 按 epoch 间隔保存
    #     if (epoch + 1) % save_period == 0 or (epoch + 1) == total_epoch:
    #         torch.save(
    #             state,
    #             os.path.join(
    #                 save_dir,
    #                 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, train_epoch_loss, val_epoch_loss)
    #             )
    #         )

    #     # best / last 权重
    #     if len(loss_history.val_loss) <= 1 or val_epoch_loss <= min(loss_history.val_loss):
    #         log('Save best model to best_epoch_weights.pth')
    #         torch.save(state, os.path.join(save_dir, "best_epoch_weights.pth"))

    #     torch.save(state, os.path.join(save_dir, "last_epoch_weights.pth"))

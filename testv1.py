# test_dataloader_eval.py
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image   # === New: 用来保存三联图和预测 mask ===

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# ====== 这里按你项目实际位置修改 ======
from utils.dataloader import SegmentationDataset_val, SegmentationDataset_train, seg_dataset_collate
from utils.utils_metrics import f_score, f_score_


def metric_np_(premask, groundtruth):
    """
    numpy 版 F1 / IoU(pixel-level)
    premask, groundtruth : 0/1 或 bool 的 numpy 数组 [H,W]
    """
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)

    if np.sum(union) == 0:
        # GT 和 pred 都没前景, 当作 perfect
        f1 = 1.0
        iou = 1.0
        return f1, iou

    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    return f1, iou

import numpy as np
from sklearn.metrics import f1_score, jaccard_score

def metric_np(premask, groundtruth):
    """
    numpy 版 F1 / IoU(pixel-level), 改为使用 sklearn:

    - 前景 = 1, 背景 = 0(二分类)
    - F1 用 sklearn.metrics.f1_score
    - IoU 用 sklearn.metrics.jaccard_score
    - 如果 GT 和 pred 都没有前景像素 => 返回 f1=1.0, iou=1.0
    """
    # 转成 numpy 数组 & 二值
    premask = np.asarray(premask)
    groundtruth = np.asarray(groundtruth)

    # 允许 bool 或 0/255 / 0/1, 统一成 0/1
    # 这里假设 "有前景" 就是值 > 0
    premask_bin = (premask > 0).astype(np.uint8)
    gt_bin      = (groundtruth > 0).astype(np.uint8)

    # 展平到一维
    y_pred = premask_bin.reshape(-1)
    y_true = gt_bin.reshape(-1)

    # 特殊情况：GT 与 pred 都完全没有前景 -> 视为 perfect
    if (y_true.max() == 0) and (y_pred.max() == 0):
        return 1.0, 1.0

    # 普通情况：用 sklearn 计算 F1 和 IoU
    f1  = f1_score(y_true, y_pred,
                   average="binary",
                   zero_division=1)   # 避免 0 除问题, 按 1.0 处理

    iou = jaccard_score(y_true, y_pred,
                        average="binary",
                        zero_division=1)

    return float(f1), float(iou)


# === New: 把 tensor 恢复成可视化的 [H,W,3] uint8 图像(大致缩放到 0-255)===
def tensor_to_img(t):
    """
    t: torch.Tensor, [C,H,W]
    不假设具体归一化方式, 只做 min-max 归一化到 [0,255]
    """
    t = t.detach().cpu().float()
    np_img = t.numpy().transpose(1, 2, 0)  # HWC
    mn, mx = np_img.min(), np_img.max()
    if mx > mn:
        np_img = (np_img - mn) / (mx - mn)
    else:
        np_img = np.zeros_like(np_img)
    np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
    return np_img


@torch.no_grad()
def evaluate_with_dataloader(model, test_loader, device, num_classes):
    """
    - 用 DataLoader 读数据(和 train 一样的 Dataset)
    - 前向走 model(...)
    - 用 train 里的 f_score 计算 pixel-level F1 (Train-style)
    - 额外算:
        * pixel-level AUC / F1 / IoU (numpy 版)
        * image-level classification ACC / F1 / AUC (GA head)
    - === New: 按 F1 分桶生成 good / normal / bad case 三联图目录 ===
    """
    model.eval()

    # --------- 1) 训练同款 F1(基于 logits + one-hot label)---------
    total_fscore = 0.0
    total_batches = 0

    # --------- 2) Pixel-level 指标(和你原 evaluate 类似)---------
    pix_auc_list = []
    pix_f1_list = []
    pix_iou_list = []

    # --------- 3) Image-level classification(GA head)---------
    img_y_true_ga = []
    img_y_score_ga = []

    # === New: F1 分桶 + 输出目录 ===
    GOOD_THRESH   = 0.70   # F1 >= 0.70  -> good
    NORMAL_THRESH = 0.40   # 0.40~0.70   -> normal
                           # < 0.40      -> bad

    cases_root = "./cases_all"
    good_dir   = os.path.join(cases_root, "good_cases")
    normal_dir = os.path.join(cases_root, "normal_cases")
    bad_dir    = os.path.join(cases_root, "bad_cases")

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    # 记录每张图的统计信息(名字 + F1 + GT 图像级标签 + GA 概率与预测)
    per_image_stats = []

    pbar = tqdm(test_loader, desc="Test (dataloader + f_score)")
    for batch_idx, batch in enumerate(pbar):
        # 兼容： (image, masks, labels) 或 (image, masks, labels, names)
        if len(batch) == 4:
            images, masks, labels, names = batch
        else:
            images, masks, labels = batch
            B = images.size(0)
            start = batch_idx * test_loader.batch_size
            names = [f"test_{start + i:06d}.png" for i in range(B)]

        images = images.to(device)
        masks  = masks.to(device)     # [N,H,W] 或 [N,1,H,W]
        labels = labels.to(device)    # [N,H,W,C+1] one-hot

        # ---------- 前向 ----------
        try:
            outputs = model(images, return_cls=True)
        except TypeError:
            outputs = model(images)

        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            seg_outputs, cls_logits = outputs   # seg_outputs: [N,C,H,W]
        else:
            seg_outputs = outputs
            cls_logits = None

        # ---------- 1) 训练同款 F1 ----------
        batch_f = f_score(seg_outputs, masks)
        # batch_f = f_score_(seg_outputs, labels)
        total_fscore += batch_f.item()
        total_batches += 1

        # ---------- 2) Pixel-level AUC / F1 / IoU ----------
        # 先拿 tampered 类的概率图(假设 class 1 是篡改)
        prob_maps = torch.softmax(seg_outputs, dim=1)[:, 1, :, :]  # [N,H,W]
        # print(prob_maps.shape)
        pred_bin  = (prob_maps > 0.5).cpu().numpy().astype(np.uint8)

        # GT: 从 masks 提出二值前景(>0 即篡改)
        if masks.ndim == 4:
            # [N,1,H,W] -> [N,H,W]
            gt_bin_t = (masks.squeeze(1) > 0)
        else:
            # [N,H,W]
            gt_bin_t = (masks > 0)
        gt_bin = gt_bin_t.cpu().numpy().astype(np.uint8)

        N = pred_bin.shape[0]

        # 如果有 GA head, 提前算好图像级概率和标签
        if cls_logits is not None:
            B = masks.size(0)
            img_labels_ga = (masks.view(B, -1).max(dim=1)[0] > 0).long()  # 只要有前景即篡改
            cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]             # 篡改概率 [B]
            img_y_true_ga.extend(img_labels_ga.cpu().numpy().tolist())
            img_y_score_ga.extend(cls_prob.cpu().numpy().tolist())
        else:
            img_labels_ga = None
            cls_prob = None

        # === New: 按样本级别处理(统计 + 分桶 + 三联图)===
        for i in range(N):
            p_bin = pred_bin[i]                 # [H,W], 0/1
            g_bin = gt_bin[i]                  # [H,W], 0/1

            # numpy 版 F1 / IoU
            f1_pix, iou_pix = metric_np(p_bin, g_bin)
            pix_f1_list.append(f1_pix)
            pix_iou_list.append(iou_pix)

            # AUC 用概率 + 0/1 GT
            g_flat = g_bin.reshape(-1)
            p_flat = prob_maps[i].cpu().numpy().reshape(-1)
            if g_flat.max() != g_flat.min():  # 排除全 0 或全 1
                try:
                    pix_auc_list.append(roc_auc_score(g_flat, p_flat))
                except ValueError:
                    pass

            # -------- 图像级 GT 标签(0=authentic, 1=tampered)--------
            img_label = 1 if g_bin.max() > 0 else 0

            # -------- GA head 概率与预测(如果有)--------
            if cls_logits is not None:
                prob_tamp = float(cls_prob[i].cpu().item())
                ga_pred = 1 if prob_tamp >= 0.5 else 0
            else:
                prob_tamp = None
                ga_pred = None

            # -------- 确定样本名 --------
            raw_name = names[i]
            base_name = os.path.splitext(os.path.basename(raw_name))[0]

            # -------- 按 F1 分桶 --------
            if f1_pix >= GOOD_THRESH:
                target_dir = good_dir
            elif f1_pix >= NORMAL_THRESH:
                target_dir = normal_dir
            else:
                target_dir = bad_dir

            os.makedirs(target_dir, exist_ok=True)

            # -------- 保存预测 mask (二值) --------
            pred_gray = (p_bin * 255).astype(np.uint8)          # [H,W]
            pred_rgb  = np.stack([pred_gray]*3, axis=-1)        # [H,W,3]
            pred_mask_path = os.path.join(target_dir, base_name + "_pred.png")
            Image.fromarray(pred_gray).save(pred_mask_path)

            # -------- 三联图 [原图 | GT mask | Pred mask] --------
            # 原图: 反归一化到 [0,255] uint8
            orig_img = tensor_to_img(images[i])                 # [H,W,3]

            # GT mask 灰度可视化
            gt_gray = (g_bin * 255).astype(np.uint8)
            gt_rgb  = np.stack([gt_gray]*3, axis=-1)

            # 对齐大小
            h = min(orig_img.shape[0], gt_rgb.shape[0], pred_rgb.shape[0])
            w = min(orig_img.shape[1], gt_rgb.shape[1], pred_rgb.shape[1])
            orig_vis = orig_img[:h, :w, :]
            gt_vis   = gt_rgb[:h, :w, :]
            pred_vis = pred_rgb[:h, :w, :]

            triplet = np.concatenate([orig_vis, gt_vis, pred_vis], axis=1)
            triplet_path = os.path.join(target_dir, base_name + "_triplet.png")
            Image.fromarray(triplet).save(triplet_path)

            # -------- 记录 per-image 统计 --------
            per_image_stats.append(
                (base_name, float(f1_pix), int(img_label), prob_tamp, ga_pred)
            )

        # 更新进度条显示一下当前 batch 的 train-style F1
        pbar.set_postfix({"F1(train_style)": batch_f.item()})

    # =================== 汇总 ===================
    # 1) train 风格的 F1(和日志里 F-score 一致)
    mean_f_train = total_fscore / max(1, total_batches)

    # 2) pixel-level(numpy 版)
    mean_pix_auc = float(np.mean(pix_auc_list)) if len(pix_auc_list) > 0 else 0.0
    mean_pix_f1  = float(np.mean(pix_f1_list))  if len(pix_f1_list) > 0 else 0.0
    mean_pix_iou = float(np.mean(pix_iou_list)) if len(pix_iou_list) > 0 else 0.0

    # 3) image-level GA head
    if len(img_y_score_ga) > 0:
        y_true_ga  = np.array(img_y_true_ga, dtype=np.int32)
        y_score_ga = np.array(img_y_score_ga, dtype=np.float32)

        try:
            img_auc_ga = roc_auc_score(y_true_ga, y_score_ga)
        except ValueError:
            img_auc_ga = 0.0

        y_pred_ga  = (y_score_ga >= 0.5).astype(np.int32)
        img_acc_ga = float((y_pred_ga == y_true_ga).mean())
        img_f1_ga  = f1_score(y_true_ga, y_pred_ga)
    else:
        img_auc_ga = img_acc_ga = img_f1_ga = 0.0

    print("\n========== Test Result (DataLoader + f_score) ==========")
    print(f"Train-style F1 (f_score): {mean_f_train:.4f}")
    print("Pixel-level:  AUC: %5.4f, F1: %5.4f, IOU: %5.4f"
          % (mean_pix_auc, mean_pix_f1, mean_pix_iou))
    print("Image-level (GA head): ACC: %5.4f, F1: %5.4f, AUC: %5.4f"
          % (img_acc_ga, img_f1_ga, img_auc_ga))
    print("=========================================================\n")

    # === New: 打印每张图的统计信息(可选)===
    print("Per-image statistics (pixel F1 & image-level prediction):")
    for fname, f1_img, label, ga_prob, ga_pred in per_image_stats:
        gt_str = "tampered(1)" if label == 1 else "authentic(0)"
        if ga_prob is not None and ga_pred is not None:
            pred_str = "tampered(1)" if ga_pred == 1 else "authentic(0)"
            print(f"{fname:30s}  F1={f1_img:.4f}  GT={gt_str}  Pred={pred_str}  p={ga_prob:.4f}")
        else:
            print(f"{fname:30s}  F1={f1_img:.4f}  GT={gt_str}  Pred=N/A  p=N/A")

    return (
        mean_f_train,
        mean_pix_auc,
        mean_pix_f1,
        mean_pix_iou,
        img_acc_ga,
        img_f1_ga,
        img_auc_ga,
    )


if __name__ == "__main__":
    from utils.test_utils_res import SegFormer_Segmentation_GA
    from utils.dataloader import SegmentationDataset_val, seg_dataset_collate
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1
    input_shape = [256, 256]
    batch_size  = 1
    num_workers = 4

    # used_weigth = r"./datasets=New_with_scSE_with_Coord_MultiStrip_Head=Normal/logs/best_epoch_weights.pth"
    used_weigth = r"./logs/ep100-loss0.014-val_loss0.154.pth"
    

    # 用你原来的封装来构建和加载模型
    segformer = SegFormer_Segmentation_GA("b2", used_weigth)
    model = segformer.net
    model.to(device)
    model.eval()

    dataset_path = r"/mnt/e/jwj/new_datasets"
    test_list    = r"/mnt/e/jwj/new_datasets/val.txt"

    with open(test_list, "r", encoding="utf-8") as f:
        test_lines = f.readlines()

    test_dataset = SegmentationDataset_train(
        test_lines,
        input_shape,
        num_classes,
        train=False,
        dataset_path=dataset_path
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate,
    )

    evaluate_with_dataloader(model, test_loader, device, num_classes)

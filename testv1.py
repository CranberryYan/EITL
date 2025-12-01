# test_dataloader_eval.py
import os
import warnings
import shlex
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score
from PIL import Image

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# ====== 这里按你项目实际位置修改 ======
from utils.dataloader import SegmentationDataset_val, SegmentationDataset_train, seg_dataset_collate
from utils.utils_metrics import f_score, f_score_


def metric_np(premask, groundtruth):
    """
    numpy 版 F1 / IoU(pixel-level), 使用 sklearn:
    - 前景 = 1, 背景 = 0
    - F1: sklearn f1_score
    - IoU: sklearn jaccard_score
    - 如果 GT 与 pred 都没有前景 => f1=1.0, iou=1.0
    """
    premask = np.asarray(premask)
    groundtruth = np.asarray(groundtruth)

    premask_bin = (premask > 0).astype(np.uint8)
    gt_bin      = (groundtruth > 0).astype(np.uint8)

    y_pred = premask_bin.reshape(-1)
    y_true = gt_bin.reshape(-1)

    if (y_true.max() == 0) and (y_pred.max() == 0):
        return 1.0, 1.0

    f1  = f1_score(y_true, y_pred, average="binary", zero_division=1)
    iou = jaccard_score(y_true, y_pred, average="binary", zero_division=1)
    return float(f1), float(iou)


def prf_from_bin(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    pred_bin / gt_bin: 0/1 uint8, shape [H,W]
    return: (P, R, F1) 按像素统计
    约定：pred=0 且 gt=0 => P=R=F1=1
    """
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_not(g)).sum())
    fn = int(np.logical_and(np.logical_not(p), g).sum())

    denom_f = 2 * tp + fp + fn
    if denom_f == 0:
        return 1.0, 1.0, 1.0, tp, fp, fn

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1   = float(2 * tp / denom_f) if denom_f > 0 else 0.0
    return prec, rec, f1, tp, fp, fn


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """
    t: [C,H,W]
    min-max 归一化到 [0,255] => uint8 [H,W,3]
    """
    t = t.detach().cpu().float()
    np_img = t.numpy().transpose(1, 2, 0)  # HWC
    mn, mx = np_img.min(), np_img.max()
    if mx > mn:
        np_img = (np_img - mn) / (mx - mn)
    else:
        np_img = np.zeros_like(np_img)
    np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
    if np_img.ndim == 2:
        np_img = np.stack([np_img]*3, axis=-1)
    if np_img.shape[2] == 1:
        np_img = np.repeat(np_img, 3, axis=2)
    return np_img


def _name_from_txt_line(line: str) -> str:
    """
    从 txt 一行解析原图文件名(不带后缀)
    支持：img mask cls（三列），也支持路径带空格（shlex）
    """
    line = line.strip()
    if not line:
        return "unknown"
    try:
        parts = shlex.split(line)
    except Exception:
        parts = line.split()
    if not parts:
        return "unknown"
    img_path = parts[0].strip('"').strip("'")
    base = os.path.basename(img_path)
    stem = os.path.splitext(base)[0]
    return stem


def _unique_path(path: str) -> str:
    """如果文件已存在，自动加 _dupN 避免覆盖"""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{root}_dup{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


@torch.no_grad()
def evaluate_with_dataloader(model, test_loader, device, num_classes, all_lines=None):
    model.eval()

    total_fscore = 0.0
    total_batches = 0

    # pixel-level lists
    pix_auc_list = []
    pix_f1_list = []
    pix_iou_list = []
    pix_p_list = []
    pix_r_list = []

    # pixel-level micro sums
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    # GA head image-level
    img_y_true_ga = []
    img_y_score_ga = []

    GOOD_THRESH   = 0.70
    NORMAL_THRESH = 0.40

    cases_root = "./cases_all"
    good_dir   = os.path.join(cases_root, "good_cases")
    normal_dir = os.path.join(cases_root, "normal_cases")
    bad_dir    = os.path.join(cases_root, "bad_cases")

    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    pbar = tqdm(test_loader, desc="Test (dataloader + f_score)")
    bs = test_loader.batch_size

    # ====== 你只需要改这两个开关 ======
    TAMPER_CH = 1                  # 预测里哪个通道代表“篡改”(0或1)
    GT_FOREGROUND_IS_ZERO = False  # True: 篡改=0；False: 篡改=255(或>127)
    PRED_THR = 0.5                 # 建议 0.5（0.9 很容易把边缘都抹掉）

    for batch_idx, batch in enumerate(pbar):
        # 兼容： (image, masks, labels) 或 (image, masks, labels, names)
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            images, masks, labels, names = batch
            if not isinstance(names, (list, tuple)):
                names = [str(names)]
        else:
            images, masks, labels = batch
            B = images.size(0)

            if all_lines is not None:
                start = batch_idx * bs
                names = []
                for i in range(B):
                    idx = start + i
                    if idx < len(all_lines):
                        names.append(_name_from_txt_line(all_lines[idx]))
                    else:
                        names.append(f"test_{idx:06d}")
            else:
                start = batch_idx * bs
                names = [f"test_{start + i:06d}" for i in range(B)]

        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        # forward
        try:
            outputs = model(images, return_cls=True)
        except TypeError:
            outputs = model(images)

        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            seg_outputs, cls_logits = outputs
        else:
            seg_outputs = outputs
            cls_logits = None

        # train-style F1
        batch_f = f_score(seg_outputs, masks)
        total_fscore += batch_f.item()
        total_batches += 1

        # ====== pixel prob map & pred bin ======
        prob_maps = torch.softmax(seg_outputs, dim=1)[:, TAMPER_CH, :, :]   # [B,H,W]
        pred_bin  = (prob_maps > PRED_THR).detach().cpu().numpy().astype(np.uint8)

        # ====== GT bin (处理 0/1 和 0/255；以及“语义反”问题) ======
        if masks.ndim == 4:
            gt_raw = masks.squeeze(1)
        else:
            gt_raw = masks

        gt_np = gt_raw.detach().cpu().numpy()
        gt_mx = gt_np.max()
        if gt_mx <= 1:
            gt_u8 = (gt_np * 255).astype(np.uint8)
        else:
            gt_u8 = gt_np.astype(np.uint8)

        if GT_FOREGROUND_IS_ZERO:
            gt_bin = (gt_u8 < 128).astype(np.uint8)
        else:
            gt_bin = (gt_u8 > 127).astype(np.uint8)

        N = pred_bin.shape[0]

        # ====== GA head（图像级） ======
        if cls_logits is not None:
            B = masks.size(0)
            img_labels_ga = (masks.view(B, -1).max(dim=1)[0] > 0).long()
            if img_labels_ga != 1:
                print("Warning: GA head but image label is 0?")
                print(names)
            cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]
            img_y_true_ga.extend(img_labels_ga.detach().cpu().numpy().tolist())
            img_y_score_ga.extend(cls_prob.detach().cpu().numpy().tolist())
        else:
            cls_prob = None

        # ====== per-image stats + save ======
        for i in range(N):
            p_bin = pred_bin[i]  # [H,W] 0/1
            g_bin = gt_bin[i]    # [H,W] 0/1

            # per-image P/R/F1 + (tp,fp,fn) for micro
            p_i, r_i, f1_i, tp_i, fp_i, fn_i = prf_from_bin(p_bin, g_bin)
            pix_p_list.append(p_i)
            pix_r_list.append(r_i)
            pix_f1_list.append(f1_i)

            micro_tp += tp_i
            micro_fp += fp_i
            micro_fn += fn_i

            # IoU（可选）
            f1_tmp, iou_pix = metric_np(p_bin, g_bin)
            pix_iou_list.append(iou_pix)

            # pixel AUC
            g_flat = g_bin.reshape(-1)
            p_flat = prob_maps[i].detach().cpu().numpy().reshape(-1)
            if g_flat.max() != g_flat.min():
                try:
                    pix_auc_list.append(roc_auc_score(g_flat, p_flat))
                except ValueError:
                    pass

            base_name = str(names[i])

            # 分档保存
            if f1_i >= GOOD_THRESH:
                target_dir = good_dir
            elif f1_i >= NORMAL_THRESH:
                target_dir = normal_dir
            else:
                target_dir = bad_dir

            # 保存 pred mask
            pred_gray = (p_bin * 255).astype(np.uint8)
            pred_mask_path = os.path.join(target_dir, base_name + "_pred.png")
            pred_mask_path = _unique_path(pred_mask_path)
            Image.fromarray(pred_gray).save(pred_mask_path)

            # 三联图
            orig_img = tensor_to_img(images[i])
            gt_gray = (g_bin * 255).astype(np.uint8)
            gt_rgb  = np.stack([gt_gray]*3, axis=-1)
            pred_rgb = np.stack([pred_gray]*3, axis=-1)

            h = min(orig_img.shape[0], gt_rgb.shape[0], pred_rgb.shape[0])
            w = min(orig_img.shape[1], gt_rgb.shape[1], pred_rgb.shape[1])

            triplet = np.concatenate(
                [orig_img[:h, :w, :], gt_rgb[:h, :w, :], pred_rgb[:h, :w, :]],
                axis=1
            )
            triplet_path = os.path.join(target_dir, base_name + "_triplet.png")
            triplet_path = _unique_path(triplet_path)
            Image.fromarray(triplet).save(triplet_path)

        pbar.set_postfix({"F1(train_style)": batch_f.item()})

    # ===== summary =====
    def safe_mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    mean_f_train = total_fscore / max(1, total_batches)

    mean_pix_auc = safe_mean(pix_auc_list)
    mean_pix_p   = safe_mean(pix_p_list)
    mean_pix_r   = safe_mean(pix_r_list)
    mean_pix_f1  = safe_mean(pix_f1_list)
    mean_pix_iou = safe_mean(pix_iou_list)

    # micro pixel P/R/F1
    micro_pix_p  = float(micro_tp / (micro_tp + micro_fp)) if (micro_tp + micro_fp) > 0 else 0.0
    micro_pix_r  = float(micro_tp / (micro_tp + micro_fn)) if (micro_tp + micro_fn) > 0 else 0.0
    micro_pix_f1 = float(2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)) if (2 * micro_tp + micro_fp + micro_fn) > 0 else 1.0

    # ===== image-level (GA head) =====
    if len(img_y_score_ga) > 0:
        y_true_ga  = np.array(img_y_true_ga, dtype=np.int32)
        y_score_ga = np.array(img_y_score_ga, dtype=np.float32)
        y_pred_ga  = (y_score_ga >= 0.5).astype(np.int32)

        tp = int(((y_pred_ga == 1) & (y_true_ga == 1)).sum())
        tn = int(((y_pred_ga == 0) & (y_true_ga == 0)).sum())
        fp = int(((y_pred_ga == 1) & (y_true_ga == 0)).sum())
        fn = int(((y_pred_ga == 0) & (y_true_ga == 1)).sum())

        img_acc_ga = float((y_pred_ga == y_true_ga).mean())
        img_prec_ga = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        img_rec_ga  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        img_f1_ga   = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0

        if len(np.unique(y_true_ga)) == 1:
            # 全是同一类
            img_auc_ga = 0.0
        else:
            img_auc_ga = roc_auc_score(y_true_ga, y_score_ga)

    else:
        img_acc_ga = img_prec_ga = img_rec_ga = img_f1_ga = img_auc_ga = 0.0

    print("\n========== Test Result (DataLoader + f_score) ==========")
    print(f"Train-style F1 (f_score): {mean_f_train:.4f}")

    print("Pixel-level (MACRO over images): "
          f"AUC={mean_pix_auc:.4f}, P={mean_pix_p:.4f}, R={mean_pix_r:.4f}, F1={mean_pix_f1:.4f}, IoU={mean_pix_iou:.4f}")
    print("Pixel-level (MICRO over all pixels): "
          f"P={micro_pix_p:.4f}, R={micro_pix_r:.4f}, F1={micro_pix_f1:.4f}")

    print("Image-level (GA head): "
          f"ACC={img_acc_ga:.4f}, P={img_prec_ga:.4f}, R={img_rec_ga:.4f}, F1={img_f1_ga:.4f}, AUC={img_auc_ga:.4f}")
    print("=========================================================\n")

    return (
        mean_f_train,
        mean_pix_auc,
        mean_pix_p, mean_pix_r, mean_pix_f1, mean_pix_iou,
        micro_pix_p, micro_pix_r, micro_pix_f1,
        img_acc_ga, img_prec_ga, img_rec_ga, img_f1_ga, img_auc_ga,
    )


if __name__ == "__main__":
    from utils.test_utils_res import SegFormer_Segmentation_GA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    input_shape = [256, 256]
    batch_size  = 1
    num_workers = 16

    # used_weigth = r"./datasets=20000_GA=True_BAM=scSE_CAF=CoordMulStrip_Head=FreqHiLoClsHead/logs/best_epoch_weights.pth"
    used_weigth = r"./datasets=20000_GA=False_BAM=scSE_CAF=CoordMulStrip_Head=FreqHiLoClsHead/logs/best_epoch_weights.pth"

    segformer = SegFormer_Segmentation_GA("b2", used_weigth)
    model = segformer.net
    model.to(device)
    model.eval()

    dataset_path = r"/mnt/e/jwj/new_datasets"
    test_list    = r"./test_txt/fused_with_allTemp_JPEG_95.txt"

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

    evaluate_with_dataloader(model, test_loader, device, num_classes, all_lines=test_lines)

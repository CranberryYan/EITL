import cv2
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from PIL import Image
from utils.test_utils_res import SegFormer_Segmentation, SegFormer_Segmentation_GA
from utils.utils import decompose, merge, rm_and_make_dir
from tqdm import tqdm
import os
import shutil
import torch
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)

    # ------ 特殊情况：GT 和 pred 都没有前景 ------
    if np.sum(union) == 0:
        # 你预测对了，而且没有任何前景要检测，可以认为是 perfect
        f1 = 1.0
        iou = 1.0
        return f1, iou

    # 正常情况
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    return f1, iou


def test_mode(dir_origin_path, dir_save_path):
    img_names = os.listdir(dir_origin_path)

    # 确保输出目录存在(且只建一次)
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path, exist_ok=True)

    img_cls_prob = {}  # 记录每张图的图像级“篡改”概率：{save_name: prob_tampered}

    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg',
                                      '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)

            # 现在 detect_image_resize 返回: seg_img, seg_pred_new, cls_prob
            _, seg_pred, cls_prob = segformer.detect_image_resize(image)

            save_name = os.path.splitext(img_name)[0] + '.png'
            seg_path  = os.path.join(dir_save_path, save_name)
            seg_pred.save(seg_path)

            # 记录图像级“篡改”概率
            if cls_prob is not None:
                # 假设 cls_prob[1] 是“篡改”类别的概率
                prob_tampered = float(cls_prob[1])
            else:
                # 如果没开 GA 分支，就用一个非常朴素的规则：mask 是否全黑
                mask_np = np.array(seg_pred, dtype=np.uint8)
                prob_tampered = 1.0 if (mask_np > 127).any() else 0.0

            img_cls_prob[save_name] = prob_tampered

    print("test_over!")
    return img_cls_prob

def find_original_image(img_root, base):
    """在 img_root 下，根据 base 名找原图，支持多种后缀。找不到就返回 None。"""
    if img_root is None:
        return None
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    for ext in exts:
        img_path = os.path.join(img_root, base + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def evaluate(path_pre, path_gt, dataset_name, record_txt, cls_prob_dict=None, img_root=None):
    # 基本存在性检查
    if not os.path.isdir(path_pre):
        print(f"[Error] 预测目录不存在: {path_pre}")
        return 0.0, 0.0, 0.0
    if not os.path.isdir(path_gt):
        print(f"[Error] GT 目录不存在: {path_gt}")
        return 0.0, 0.0, 0.0

    flist = sorted(os.listdir(path_pre))

    # 像素级指标
    auc_list, f1_list, iou_list = [], [], []

    # 用于记录每张图的 F1、GT 是否篡改、GA 概率和 GA 预测结果
    # 元素格式: (filename, f1_pix, img_label, ga_prob, ga_pred)
    per_image_stats = []

    # 图像级指标（只基于 GA head 概率）
    img_y_true_ga  = []   # 图像级 GT: 0=原图(全黑mask), 1=篡改
    img_y_score_ga = []   # GA head 输出的篡改概率

    # ========= F1 分桶阈值 & 结果保存目录 =========
    GOOD_THRESH   = 0.70   # F1 >= 0.70  -> good
    NORMAL_THRESH = 0.40   # 0.40~0.70   -> normal
                           # < 0.40      -> bad

    cases_root = "./cases_all"
    good_dir   = os.path.join(cases_root, "good_cases")
    normal_dir = os.path.join(cases_root, "normal_cases")
    bad_dir    = os.path.join(cases_root, "bad_cases")

    for d in [good_dir, normal_dir, bad_dir]:
        os.makedirs(d, exist_ok=True)

    for file in tqdm(flist):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        try:
            pre_path = os.path.join(path_pre, file)

            # GT 命名: xxx.png -> xxx_gt.png
            base    = os.path.splitext(file)[0]
            gt_name = base + '_gt.png'
            gt_path = os.path.join(path_gt, gt_name)

            pre = cv2.imread(pre_path)
            gt  = cv2.imread(gt_path)

            if pre is None or gt is None:
                print(f"[Warn] 读图失败: pre={pre_path}, gt={gt_path}")
                continue

            H, W, C    = pre.shape
            Hg, Wg, Cg = gt.shape

            if H != Hg or W != Wg:
                gt = cv2.resize(gt, (W, H))
                gt[gt > 70] = 255
                gt[gt <= 70] = 0

            # ---------- 像素级 AUC / F1 / IoU ----------
            # AUC 用连续值
            if np.max(gt) != np.min(gt):  # 排除全黑/全白图
                y_true_pix  = (gt.reshape(-1) / 255).astype('int')
                y_score_pix = pre.reshape(-1) / 255.
                auc_list.append(roc_auc_score(y_true_pix, y_score_pix))

            # F1 / IoU 用二值化 mask
            pre_bin = pre.copy()
            gt_bin  = gt.copy()
            pre_bin[pre_bin > 70] = 255
            pre_bin[pre_bin <= 70] = 0
            gt_bin[gt_bin > 70] = 255
            gt_bin[gt_bin <= 70] = 0

            f1_pix, iou_pix = metric(pre_bin / 255, gt_bin / 255)
            f1_list.append(f1_pix)
            iou_list.append(iou_pix)

            # ---------- 图像级 GT：由 GT mask 决定 ----------
            # GT mask 全黑 -> 原图(0)，否则篡改(1)
            gt_has_fg = (gt_bin > 127).any()
            img_label = 1 if gt_has_fg else 0  # GT 图像级标签

            # ---------- GA head 的图像级概率 & 预测 ----------
            ga_prob = None
            ga_pred = None
            if cls_prob_dict is not None:
                key = file  # test_mode 里保存的就是 base+'.png'
                if key in cls_prob_dict:
                    ga_prob = float(cls_prob_dict[key])
                    # 阈值 0.5：>=0.5 预测为篡改
                    ga_pred = 1 if ga_prob >= 0.5 else 0

                    img_y_true_ga.append(img_label)
                    img_y_score_ga.append(ga_prob)

            # 记录当前图片的统计信息
            per_image_stats.append(
                (file, float(f1_pix), int(img_label), ga_prob, ga_pred)
            )

            # ========== 按 F1 分桶 & 复制预测 mask ==========
            if f1_pix >= GOOD_THRESH:
                target_dir = good_dir
            elif f1_pix >= NORMAL_THRESH:
                target_dir = normal_dir
            else:
                target_dir = bad_dir

            # 复制预测 mask 到对应目录
            dst_mask_path = os.path.join(target_dir, file)
            try:
                shutil.copy(pre_path, dst_mask_path)
            except Exception as e:
                print(f"[Warn] 拷贝预测 mask 失败: {pre_path} -> {dst_mask_path}, err={e}")

            # ========== 生成 [原图 | GT mask | 预测 mask] 三联图 ==========
            # 1) 找原图
            orig = None
            if img_root is not None:
                orig_path = find_original_image(img_root, base)
                if orig_path is not None:
                    orig = cv2.imread(orig_path)

            # 如果找不到原图，就用黑图占位，保证形状一致
            if orig is None:
                orig = np.zeros_like(pre)

            # 统一大小到预测 mask 的 H, W
            if orig.shape[0] != H or orig.shape[1] != W:
                orig_vis = cv2.resize(orig, (W, H))
            else:
                orig_vis = orig

            if gt_bin.shape[0] != H or gt_bin.shape[1] != W:
                gt_vis = cv2.resize(gt_bin, (W, H))
            else:
                gt_vis = gt_bin

            pre_vis = pre_bin  # 用二值化后的预测 mask 作为可视化

            # 横向拼接: [原图 | GT | 预测]
            triplet = np.concatenate([orig_vis, gt_vis, pre_vis], axis=1)
            triplet_path = os.path.join(target_dir, base + "_triplet.png")
            cv2.imwrite(triplet_path, triplet)

        except Exception as e:
            print(f"[Error] file={file}, err={e}")

    if len(f1_list) == 0:
        print(f"[Error] 没有成功评估的样本, 检查路径和命名是否匹配.")
        return 0.0, 0.0, 0.0

    # ---------- 像素级指标 ----------
    mean_auc = float(np.mean(auc_list)) if len(auc_list) > 0 else 0.0
    mean_f1  = float(np.mean(f1_list))
    mean_iou = float(np.mean(iou_list))

    print(dataset_name)
    print('Pixel-level: AUC: %5.4f, F1: %5.4f, IOU: %5.4f'
          % (mean_auc, mean_f1, mean_iou))

    # ---------- 图像级（GA head）指标 ----------
    if len(img_y_score_ga) > 0:
        y_true_ga  = np.array(img_y_true_ga, dtype=np.int32)
        y_score_ga = np.array(img_y_score_ga, dtype=np.float32)

        # AUC：0 vs 1 的曲线
        try:
            img_auc_ga = roc_auc_score(y_true_ga, y_score_ga)
        except ValueError:
            img_auc_ga = 0.0  # 只有单一类别时 AUC 不可定义，这里给个 0.0

        # 阈值 0.5 做二分类
        y_pred_ga  = (y_score_ga >= 0.5).astype(np.int32)
        img_acc_ga = float((y_pred_ga == y_true_ga).mean())
        img_f1_ga  = f1_score(y_true_ga, y_pred_ga)
    else:
        img_auc_ga = img_acc_ga = img_f1_ga = 0.0

    print('Image-level (GA head): ACC: %5.4f, F1: %5.4f, AUC: %5.4f'
          % (img_acc_ga, img_f1_ga, img_auc_ga))

    # ============ 打印每张图的 文件名 / F1 / GT / 预测结果 ============
    print("\nPer-image statistics (pixel F1 & image-level prediction):")
    for fname, f1_img, label, ga_prob, ga_pred in per_image_stats:
        # GT 字符串
        gt_str = "tampered(1)" if label == 1 else "authentic(0)"

        # 预测字符串（如果没有 GA 概率，则标记为 N/A）
        if ga_prob is not None and ga_pred is not None:
            pred_str = "tampered(1)" if ga_pred == 1 else "authentic(0)"
            print(f"{fname:30s}  F1={f1_img:.4f}  GT={gt_str}  Pred={pred_str}  p={ga_prob:.4f}")
        else:
            print(f"{fname:30s}  F1={f1_img:.4f}  GT={gt_str}  Pred=N/A  p=N/A")

    # 写入日志
    with open(record_txt, "a") as f:
        f.writelines(dataset_name + "\n")
        f.writelines('Pixel-level:  AUC: %5.4f, F1: %5.4f, IOU: %5.4f\n'
                     % (mean_auc, mean_f1, mean_iou))
        f.writelines('Image-level (GA head): ACC: %5.4f, F1: %5.4f, AUC: %5.4f\n'
                     % (img_acc_ga, img_f1_ga, img_auc_ga))
        f.writelines("Per-image statistics (pixel F1 & image-level prediction):\n")
        for fname, f1_img, label, ga_prob, ga_pred in per_image_stats:
            gt_str = "tampered(1)" if label == 1 else "authentic(0)"
            if ga_prob is not None and ga_pred is not None:
                pred_str = "tampered(1)" if ga_pred == 1 else "authentic(0)"
                f.writelines(
                    f"{fname}\tF1={f1_img:.4f}\tGT={gt_str}\tPred={pred_str}\tp={ga_prob:.4f}\n"
                )
            else:
                f.writelines(
                    f"{fname}\tF1={f1_img:.4f}\tGT={gt_str}\tPred=N/A\tp=N/A\n"
                )
        f.writelines("\n")

    return mean_auc, mean_f1, mean_iou


if __name__ == "__main__":
    used_weigth = r"./logs/best_epoch_weights.pth"
    segformer = SegFormer_Segmentation_GA("b2", used_weigth)

    record_txt = r"./test_out/evaluate_result.txt"
    record_dir = os.path.dirname(record_txt)
    if record_dir != '' and not os.path.exists(record_dir):
        os.makedirs(record_dir, exist_ok=True)

    with open(record_txt, "a") as f:
        f.writelines(str(used_weigth) + "\n")

    test_path = r'/mnt/e/jwj/test_data/Images/'
    save_path = r'/mnt/e/jwj/test_data/out/'
    path_gt   = r'/mnt/e/jwj/test_data/Mask0255/'

    rm_and_make_dir(save_path)

    cls_prob_dict = test_mode(test_path, save_path)

    # 多传入 img_root=test_path
    auc, f1, iou = evaluate(save_path, path_gt, "EITL-Test",
                            record_txt, cls_prob_dict, img_root=test_path)

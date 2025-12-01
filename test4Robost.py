import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score

from utils.utils import preprocess_input
from utils.utils_metrics import f_score
from utils.test_utils_res import SegFormer_Segmentation_GA


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _clean(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def _safe_stem(rel_name: str) -> str:
    rel_name = _clean(rel_name)
    stem = os.path.splitext(rel_name)[0]
    return stem.replace("/", "__")


class RobostTxtDataset(Dataset):
    """
    txt 每行： ./Robost_Jpeg_80/.../(xxx).jpg
    mask 在： ./robost_mask/(xxx).png  (只取 basename 对齐)
    """
    def __init__(self, lines, input_shape, num_classes, dataset_path, mask_root="./robost_mask"):
        self.lines = [_clean(ln) for ln in lines if ln.strip()]
        self.input_shape = input_shape  # [H,W]
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.mask_root = mask_root

        self.jpeg_root = os.path.join(self.dataset_path, "JPEGImages")

    def __len__(self):
        return len(self.lines)

    def _resolve_img(self, rel: str) -> str:
        # rel 已经是 Robost_Jpeg_80/.../xxx.jpg
        # 若 rel 不带后缀，默认补 .jpg，并尝试其它后缀兜底
        root, ext = os.path.splitext(rel)
        if ext.lower() in IMG_EXTS:
            cand = os.path.join(self.jpeg_root, rel)
            if os.path.isfile(cand):
                return cand
            raise FileNotFoundError(f"Image not found: {cand}")
        else:
            cand = os.path.join(self.jpeg_root, rel + ".jpg")
            if os.path.isfile(cand):
                return cand
            for e in IMG_EXTS:
                cand = os.path.join(self.jpeg_root, rel + e)
                if os.path.isfile(cand):
                    return cand
            raise FileNotFoundError(f"Image not found: {os.path.join(self.jpeg_root, rel)} (tried many exts)")

    def _resolve_mask(self, rel_img: str) -> str:
        # 只用 basename 做匹配
        base = os.path.splitext(os.path.basename(rel_img))[0]  # (GalaxyN3)122_Sony...
        mask_path = os.path.join(self.mask_root, base + ".png")
        return mask_path  # 允许不存在，外面会兜底

    def __getitem__(self, idx):
        rel = self.lines[idx]  # Robost_Jpeg_80/.../xxx.jpg
        img_path = self._resolve_img(rel)
        mask_path = self._resolve_mask(rel)

        img = Image.open(img_path).convert("RGB")

        if os.path.isfile(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # 没有 GT 时，给全 0 mask
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")

        # resize 到 input_shape，避免 batch 堆叠失败
        H, W = self.input_shape
        img = img.resize((W, H), Image.BICUBIC)
        mask = mask.resize((W, H), Image.NEAREST)

        # img -> [C,H,W]
        img_np = np.array(img, dtype=np.float64)
        img_np = preprocess_input(img_np)
        img_np = np.transpose(img_np, (2, 0, 1)).astype(np.float32)

        # mask -> [H,W]，二分类：>0 即篡改
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        if self.num_classes == 1:
            mask_np = (mask_np > 0).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.int64)
            mask_np[mask_np >= self.num_classes] = self.num_classes

        h, w = mask_np.shape[:2]
        seg_labels = np.eye(self.num_classes + 1)[mask_np.reshape(-1)]
        seg_labels = seg_labels.reshape((h, w, self.num_classes + 1)).astype(np.float32)

        # 返回 rel 作为 name（保存用）
        return img_np, mask_np, seg_labels, rel


def collate_fn(batch):
    images, masks, seg_labels, names = [], [], [], []
    for img, mask, lab, nm in batch:
        images.append(img)
        masks.append(mask)
        seg_labels.append(lab)
        names.append(nm)
    images = torch.from_numpy(np.array(images)).float()
    masks = torch.from_numpy(np.array(masks)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).float()
    return images, masks, seg_labels, names


def tensor_to_img(t):
    t = t.detach().cpu().float()
    np_img = t.numpy().transpose(1, 2, 0)
    mn, mx = np_img.min(), np_img.max()
    if mx > mn:
        np_img = (np_img - mn) / (mx - mn)
    else:
        np_img = np.zeros_like(np_img)
    np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
    return np_img


def metric_np(pred_bin, gt_bin):
    y_pred = pred_bin.reshape(-1).astype(np.uint8)
    y_true = gt_bin.reshape(-1).astype(np.uint8)

    if (y_true.max() == 0) and (y_pred.max() == 0):
        return 1.0, 1.0

    f1v = f1_score(y_true, y_pred, average="binary", zero_division=1)
    iou = jaccard_score(y_true, y_pred, average="binary", zero_division=1)
    return float(f1v), float(iou)


@torch.no_grad()
def evaluate(model, loader, device, save_cases=True):
    model.eval()

    pix_auc_list, pix_f1_list, pix_iou_list = [], [], []
    total_f = 0.0
    total_batches = 0

    GOOD_THRESH = 0.70
    NORMAL_THRESH = 0.40

    cases_root = "./cases_all"
    good_dir = os.path.join(cases_root, "good_cases")
    normal_dir = os.path.join(cases_root, "normal_cases")
    bad_dir = os.path.join(cases_root, "bad_cases")
    if save_cases:
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)

    pbar = tqdm(loader, desc="Eval")
    for images, masks, seg_labels, names in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # forward
        try:
            out = model(images, return_cls=True)
        except TypeError:
            out = model(images)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            seg_logits, cls_logits = out
        else:
            seg_logits, cls_logits = out, None

        batch_f = f_score(seg_logits, masks)
        total_f += float(batch_f.item())
        total_batches += 1

        prob = torch.softmax(seg_logits, dim=1)[:, 1]  # [B,H,W]
        pred_bin = (prob > 0.5).detach().cpu().numpy().astype(np.uint8)
        gt_bin = (masks > 0).detach().cpu().numpy().astype(np.uint8)

        B = pred_bin.shape[0]
        for i in range(B):
            f1_pix, iou_pix = metric_np(pred_bin[i], gt_bin[i])
            pix_f1_list.append(f1_pix)
            pix_iou_list.append(iou_pix)

            g_flat = gt_bin[i].reshape(-1)
            p_flat = prob[i].detach().cpu().numpy().reshape(-1)
            if g_flat.max() != g_flat.min():
                try:
                    pix_auc_list.append(roc_auc_score(g_flat, p_flat))
                except ValueError:
                    pass

            if save_cases:
                stem = _safe_stem(names[i])

                if f1_pix >= GOOD_THRESH:
                    out_dir = good_dir
                elif f1_pix >= NORMAL_THRESH:
                    out_dir = normal_dir
                else:
                    out_dir = bad_dir

                pred_gray = (pred_bin[i] * 255).astype(np.uint8)
                Image.fromarray(pred_gray).save(os.path.join(out_dir, stem + "_pred.png"))

                orig = tensor_to_img(images[i])
                gt_gray = (gt_bin[i] * 255).astype(np.uint8)
                gt_rgb = np.stack([gt_gray] * 3, axis=-1)
                pred_rgb = np.stack([pred_gray] * 3, axis=-1)

                h = min(orig.shape[0], gt_rgb.shape[0], pred_rgb.shape[0])
                w = min(orig.shape[1], gt_rgb.shape[1], pred_rgb.shape[1])
                triplet = np.concatenate([orig[:h, :w], gt_rgb[:h, :w], pred_rgb[:h, :w]], axis=1)
                Image.fromarray(triplet).save(os.path.join(out_dir, stem + "_triplet.png"))

        pbar.set_postfix({"F1(train_style)": float(batch_f.item())})

    mean_train_style_f = total_f / max(1, total_batches)
    mean_pix_auc = float(np.mean(pix_auc_list)) if len(pix_auc_list) else 0.0
    mean_pix_f1 = float(np.mean(pix_f1_list)) if len(pix_f1_list) else 0.0
    mean_pix_iou = float(np.mean(pix_iou_list)) if len(pix_iou_list) else 0.0

    print("\n========== Eval Result ==========")
    print(f"Train-style F1 (f_score): {mean_train_style_f:.4f}")
    print(f"Pixel-level: AUC={mean_pix_auc:.4f}  F1={mean_pix_f1:.4f}  IoU={mean_pix_iou:.4f}")
    print("================================\n")
    return mean_train_style_f, mean_pix_auc, mean_pix_f1, mean_pix_iou


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 你需要改的路径 =====
    dataset_path = r"/mnt/e/jwj/new_datasets"
    test_list = r"./sampled_files_jpeg_95.txt"
    mask_root = r"./robost_mask"   # 你的 GT mask 文件夹
    used_weight = r"./datasets=20000_GA=True_BAM=scSE_CAF=CoordMulStrip_Head=FreqHiLoClsHead/logs/best_epoch_weights.pth"
    # =========================

    input_shape = [256, 256]
    num_classes = 1
    batch_size = 1
    num_workers = 16

    with open(test_list, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    ds = RobostTxtDataset(
        lines=lines,
        input_shape=input_shape,
        num_classes=num_classes,
        dataset_path=dataset_path,
        mask_root=mask_root
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    segformer = SegFormer_Segmentation_GA("b2", used_weight)
    model = segformer.net.to(device).eval()

    evaluate(model, loader, device, save_cases=True)

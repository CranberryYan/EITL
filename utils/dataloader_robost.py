# utils/dataloader_robost.py
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import preprocess_input


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _clean_path(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def _try_candidates(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def _resolve_image_path(dataset_path: str, img_str: str) -> Tuple[str, str]:
    """
    返回 (abs_img_path, rel_under_jpegimages)
    img_str 可为绝对路径或相对路径；可带后缀或不带后缀
    """
    img_str = _clean_path(img_str)

    # 绝对路径：直接用
    if os.path.isabs(img_str):
        if not os.path.isfile(img_str):
            raise FileNotFoundError(f"Image not found (abs): {img_str}")
        # 尽量推断 rel（如果它在 JPEGImages 下）
        jpeg_root = os.path.join(dataset_path, "JPEGImages").replace("\\", "/")
        abs_norm = img_str.replace("\\", "/")
        if abs_norm.startswith(jpeg_root + "/"):
            rel = abs_norm[len(jpeg_root) + 1:]
        else:
            rel = os.path.basename(abs_norm)
        return img_str, rel

    jpeg_root = os.path.join(dataset_path, "JPEGImages")

    root, ext = os.path.splitext(img_str)
    if ext.lower() in IMG_EXTS:
        cands = [os.path.join(jpeg_root, img_str)]
        rel = img_str
    else:
        # 默认补 .jpg，再兜底尝试其它后缀
        cands = [os.path.join(jpeg_root, img_str + ".jpg")]
        cands += [os.path.join(jpeg_root, img_str + e) for e in IMG_EXTS]
        rel = img_str + ".jpg"

    hit = _try_candidates(cands)
    if hit is None:
        raise FileNotFoundError(
            f"Image not found: rel='{img_str}' under {jpeg_root}\nTried: " + "\n".join(cands[:6])
        )

    # 反推 rel（用于导出名字/找 mask）
    rel = os.path.relpath(hit, jpeg_root).replace("\\", "/")
    return hit, rel


def _resolve_mask_path(dataset_path: str, img_rel_under_jpeg: str, mask_str: Optional[str]) -> Optional[str]:
    """
    规则：
    - 若 mask_str 给了：按绝对/相对方式解析，支持带/不带后缀
    - 若 mask_str 没给：用 img_rel_under_jpeg 推导 SegmentationClass 下同结构文件，优先 .png
    """
    seg_root = os.path.join(dataset_path, "SegmentationClass")

    # 显式给 mask
    if mask_str is not None and mask_str.strip():
        mask_str = _clean_path(mask_str)

        if os.path.isabs(mask_str):
            return mask_str if os.path.isfile(mask_str) else None

        root, ext = os.path.splitext(mask_str)
        if ext.lower() in MASK_EXTS:
            cands = [os.path.join(seg_root, mask_str)]
        else:
            cands = [os.path.join(seg_root, mask_str + ".png")]
            cands += [os.path.join(seg_root, mask_str + e) for e in MASK_EXTS]

        return _try_candidates(cands)

    # 推导 mask：保持子目录结构一致
    img_rel = _clean_path(img_rel_under_jpeg)
    stem = os.path.splitext(img_rel)[0]  # Robost_Jpeg_80/.../93956
    cands = [os.path.join(seg_root, stem + ".png")]
    cands += [os.path.join(seg_root, stem + e) for e in MASK_EXTS]
    return _try_candidates(cands)


def _resize_pair(img: Image.Image, mask: Image.Image, size_hw: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    """
    size_hw: (H, W)
    """
    H, W = size_hw
    img = img.resize((W, H), Image.BICUBIC)
    mask = mask.resize((W, H), Image.NEAREST)
    return img, mask


class RobostSegDataset(Dataset):
    """
    annotation_lines 每行支持：
      1) 只有 img 相对路径/绝对路径（你现在的情况）
         e.g. ./Robost_Jpeg_80/copymove/fake/93956.jpg
      2) img mask（相对或绝对）
         e.g. ./xxx.jpg ./yyy.png
      3) img mask class_name ...（多余字段忽略）
    """
    def __init__(self, annotation_lines, input_shape, num_classes, dataset_path, train=False):
        super().__init__()
        self.lines = [ln.strip() for ln in annotation_lines if ln.strip()]
        self.input_shape = input_shape  # [H,W]
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.train = train

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        parts = line.split()

        img_str = parts[0]
        mask_str = parts[1] if len(parts) >= 2 else None

        img_path, img_rel = _resolve_image_path(self.dataset_path, img_str)
        mask_path = _resolve_mask_path(self.dataset_path, img_rel, mask_str)

        img = Image.open(img_path).convert("RGB")

        if mask_path is not None and os.path.isfile(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # 找不到 mask（比如真实图），生成全 0 mask
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")

        # resize 到网络输入大小，避免 batch 堆叠失败
        img, mask = _resize_pair(img, mask, (self.input_shape[0], self.input_shape[1]))

        # image -> [C,H,W]
        img_np = np.array(img, dtype=np.float64)
        img_np = preprocess_input(img_np)
        img_np = np.transpose(img_np, (2, 0, 1))

        # mask -> [H,W] int
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        # 二分类：把 >0 视为篡改
        if self.num_classes == 1:
            mask_np = (mask_np > 0).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.int64)
            mask_np[mask_np >= self.num_classes] = self.num_classes

        h, w = mask_np.shape[:2]
        seg_labels = np.eye(self.num_classes + 1)[mask_np.reshape(-1)]
        seg_labels = seg_labels.reshape((h, w, self.num_classes + 1)).astype(np.float32)

        # 额外返回 name（用于保存结果、对齐 txt）
        name = img_rel  # Robost_Jpeg_80/.../93956.jpg
        return img_np, mask_np, seg_labels, name


def robost_collate_fn(batch):
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

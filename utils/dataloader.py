import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor
# import random
import copy
class SegmentationDataset_train(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset_train, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.IMG_EXTS  = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]
        self.MASK_EXTS = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"]

    def __len__(self):
        return self.length

    def _find_existing_path(self, base_path_no_ext: str, exts):
        """给定不带后缀的 base，按 exts 依次尝试找到真实存在的文件路径。"""
        for e in exts:
            p = base_path_no_ext + e
            if os.path.isfile(p):
                return p
        return None

    def _resolve_image_path(self, dataset_root: str, name: str):
        """
        name 可以是：
        1) 相对路径：copymove/fake/52061   或 authentic/train2014resize_256/xxx
        2) 绝对路径：/mnt/e/datasets/.../copymove/fake/52061
        3) 带后缀或不带后缀均可
        """
        name = name.strip().strip("\n").strip()

        # 先拼成绝对/完整路径
        p = name if os.path.isabs(name) else os.path.join(dataset_root, name)

        root, ext = os.path.splitext(p)
        if ext.lower() in self.IMG_EXTS and os.path.isfile(p):
            return p  # 已经带后缀且存在
        # 不带后缀：尝试各种后缀
        img_path = self._find_existing_path(root, self.IMG_EXTS)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for name={name} (tried: {root} + {self.IMG_EXTS})")
        return img_path

    def _is_authentic(self, path_str: str):
        s = path_str.replace("\\", "/")
        # 你给的 authentic 示例路径含 train2014resize_256
        return ("/authentic/" in s) or ("train2014resize_256" in s)

    def _derive_mask_path_from_fake(self, img_path: str):
        """
        fake:  .../<type>/fake/<name>.<ext>
        mask:  .../<type>/mask/<name>.<mask_ext>
        """
        s = img_path.replace("\\", "/")
        root, _ = os.path.splitext(s)

        # 优先按 '/fake/' -> '/mask/' 替换
        if "/fake/" in root:
            mask_root = root.replace("/fake/", "/mask/")
        else:
            # 兜底：如果上级目录叫 fake，就换成 mask
            parts = root.split("/")
            if len(parts) >= 2 and parts[-2] == "fake":
                parts[-2] = "mask"
            mask_root = "/".join(parts)

        mask_path = self._find_existing_path(mask_root, self.MASK_EXTS)
        return mask_path  # 可能为 None

    def __getitem__(self, index):
        """
        支持两种 annotation 格式: 
        1) 新格式(推荐): <img_abs_path> <mask_abs_path> <class_name>
            例如: /mnt/e/jwj/new_datasets/Image/copymove_64302.jpg /mnt/e/jwj/new_datasets/Mask/copymove_64302.png copymove
        2) 旧格式(兼容): name
            仍然从 dataset_path/JPEGImages 和 dataset_path/SegmentationClass 下面去拼路径
        """
        line = self.annotation_lines[index].strip()
        parts = line.split()

        # ============= 新格式: abs img path + abs mask path (+ class_name) =============
        if len(parts) >= 2 and os.path.isabs(parts[0]) and os.path.isabs(parts[1]):
            img_path  = parts[0]
            mask_path = parts[1]
            cls_name  = parts[2] if len(parts) > 2 else None  # 目前用不到，可以先存起来

            # 读图像
            img = Image.open(img_path)
            # 保证是 RGB（原代码里后面按 3 通道来处理）
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 读 mask
            mask = Image.open(mask_path)
            # 通常用单通道即可，后续 np.array(mask) 就是 [H,W]
            if mask.mode != "L":
                mask = mask.convert("L")

        # ============= 兼容旧格式: 只有 name 的情况 =============
        else:
            # ===================== 在 __getitem__ 里用这段 =====================
            name = line.strip()

            img_path = self._resolve_image_path(self.dataset_path, name)
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # authentic：没有 mask -> 全黑
            if self._is_authentic(img_path):
                mask = Image.new("L", img.size, 0)
            else:
                mask_path = self._derive_mask_path_from_fake(img_path)
                if mask_path is None:
                    # 如果列表里混进了 fake 但确实找不到 mask，也给全黑，避免训练/评测崩
                    mask = Image.new("L", img.size, 0)
                else:
                    mask = Image.open(mask_path)
                    if mask.mode != "L":
                        mask = mask.convert("L")
        # ============= 后续处理: 保持你原来的逻辑不变 =============

        # Data augmentation
        # img, mask = self.get_random_data(img, mask, self.input_shape, random_flag=self.train)

        # img: [H, W, 3] -> 预处理 + CHW
        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])

        # mask: [H, W]，值范围一般是 0, 255 或 0,1,...
        mask = np.array(mask)

        # 保证在 [0, num_classes] 范围内
        mask[mask >= self.num_classes] = self.num_classes

        # 用 mask 的真实高宽，而不是死用 self.input_shape
        h, w = mask.shape[:2]

        seg_labels = np.eye(self.num_classes + 1)[mask.reshape(-1)]
        seg_labels = seg_labels.reshape((h, w, self.num_classes + 1))

        return img, mask, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random_flag=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random_flag:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw,nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label
        # #resize
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # RandomCrop
        if (nw < w) or (nh < h):
            dx = np.random.randint(0, max(0, nw - w)+1)
            dy = np.random.randint(0, max(0, nh - h)+1)
            new_w = max(w, nw)
            new_h = max(h, nh)
            new_image = Image.new('RGB', (new_w, new_h), (128, 128, 128))
            new_label = Image.new('L', (new_w, new_h), (0))
            new_image.paste(image, (dx, dy))
            new_label.paste(label, (dx, dy))
            image = copy.deepcopy(new_image)
            label = new_label

        x = np.random.randint(0, max(0, nw - w)+1)
        y = np.random.randint(0, max(0, nh - h)+1)
        image1 = np.array(image)
        label1 = np.array(label)

        # print("hi1", np.array(image).shape)
        # image = image1[y:y + 512, x:x + 512]
        # label = label1[y:y + 512, x:x + 512]
        image = image1[y:y + 256, x:x + 256]
        label = label1[y:y + 256, x:x + 256]

        # RandomFlip
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        else:
            image = image
            label = label

        # AddGaussianNoise
        h, w, c = image.shape
        noise = np.random.normal(0, 30, (h, w, c))
        image_data = np.clip(image + noise, 0, 255).astype(np.uint8)

        # GaussianBlur
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # Pil_jpg
        open_cv_image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', open_cv_image)
        decoded_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        image_data = cv2.resize(decoded_image, (image_data.shape[1], image_data.shape[0]))


        return image_data, label

class SegmentationDataset_val(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset_val, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index].strip()
        parts = annotation_line.split()

        # ===================== 1. 解析 train.txt =====================
        if len(parts) == 1:
            # ===== 老版: 只有 name =====
            name = parts[0]

            img_dir = os.path.join(self.dataset_path, "JPEGImages")
            seg_dir = os.path.join(self.dataset_path, "SegmentationClass")

            # 尝试多种后缀找图像
            img_path = None
            for ext in [".jpg", ".png", ".tif", ".bmp", ".jpeg"]:
                p = os.path.join(img_dir, name + ext)
                if os.path.isfile(p):
                    img_path = p
                    break
            if img_path is None:
                raise FileNotFoundError(f"Image not found for base name '{name}' in {img_dir}")

            # 尝试多种后缀找 mask
            mask_path = None
            for ext in [".png", ".tif", ".jpg", ".bmp", ".jpeg"]:
                p = os.path.join(seg_dir, name + ext)
                if os.path.isfile(p):
                    mask_path = p
                    break
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for base name '{name}' in {seg_dir}")

        else:
            # ===== 新版: img_abs mask_abs class_name ... =====
            img_path  = parts[0]
            mask_path = parts[1]
            # 第三个是类别名（authentic/splice/…），这里暂时用不到，可以按需存下

        # ===================== 2. 读图 =====================
        # 图像
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # mask 作为单通道
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        # ===================== 3. 数据增强 =====================
        # img, mask = self.get_random_data(img, mask, self.input_shape, random_flag=self.train)

        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])

        mask = np.array(mask)
        # 万一 get_random_data 返回的是 H×W×1，压掉多余通道
        if mask.ndim == 3:
            mask = mask[..., 0]

        # 限制在 [0, num_classes]
        mask[mask >= self.num_classes] = self.num_classes

        # ===================== 4. 用 mask 实际大小生成 seg_labels =====================
        h, w = mask.shape[:2]

        seg_labels = np.eye(self.num_classes + 1)[mask.reshape(-1)]
        seg_labels = seg_labels.reshape((h, w, self.num_classes + 1))

        return img, mask, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random_flag=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random_flag:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw,nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label
        # #resize
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # RandomCrop
        if (nw < w) or (nh < h):
            dx = np.random.randint(0, max(0, nw - w))
            dy = np.random.randint(0, max(0, nh - h))
            new_w = max(w, nw)
            new_h = max(h, nh)
            new_image = Image.new('RGB', (new_w, new_h), (128, 128, 128))
            new_label = Image.new('L', (new_w, new_h), (0))
            new_image.paste(image, (dx, dy))
            new_label.paste(label, (dx, dy))
            image = copy.deepcopy(new_image)
            label = new_label

        x = np.randint(0, max(0, nw - w))
        y = np.random.randint(0, max(0, nh - h))
        image = np.array(image)
        label = np.array(label)

        # print("hi1", np.array(image).shape)
        # image_data = image[y:y + 512, x:x + 512]
        # label = label[y:y + 512, x:x + 512]
        image_data = image[y:y + 256, x:x + 256]
        label = label[y:y + 256, x:x + 256]

        return image_data, label

def seg_dataset_collate(batch):
    images = []
    masks = []
    seg_labels = []
    for img, mask, labels in batch:
        images.append(img)
        masks.append(mask)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    masks = torch.from_numpy(np.array(masks)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, masks, seg_labels

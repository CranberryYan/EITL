import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.EITLnet import SegFormer, SegFormer_GA
from utils.utils import cvtColor, preprocess_input, resize_image

class SegFormer_Segmentation(object):
    _defaults = {
        "model_path": "",  # 和save_dir一致
        "num_classes": 2,
        "phi": "b2",
        "input_shape": [256, 256],
        "cuda": True,
    }

    # 初始化SegFormer
    def __init__(self,phi,path,**kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes == 2:
            self.colors = [(0, 0, 0), (255, 255, 255)]
        if path!='':
            self.model_path=path
        if phi!='':
            self.phi=phi
        self.generate()

        # show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = SegFormer(num_classes=self.num_classes, phi=self.phi, dual=True, pretrained=False)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net.load_state_dict(torch.load(self.model_path, map_location=device,)['state_dict'],strict=False)
        self.net = self.net.eval()
        # print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image_resize(self, image):
        image = cvtColor(image)

        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            seg_pred = pr[:, :, 1]
            seg_pred_new = seg_pred.reshape(orininal_h, orininal_w)
            seg_pred_new = Image.fromarray(np.uint8(seg_pred_new * 255))

            pr = pr.argmax(axis=-1)

        seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))

        return image, seg_pred_new

    def detect_image_noresize(self, image):
        image = cvtColor(image)
        # 对输入图像进行一个备份, 后面用于绘图
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 添加batch_size维度
        image = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            seg_pred = pr[:, :, 1]
            seg_pred_new = seg_pred.reshape(orininal_h, orininal_w)
            seg_pred_new = Image.fromarray(np.uint8(seg_pred_new * 255))

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return image, seg_pred_new

class SegFormer_Segmentation_GA(object):
    _defaults = {
        "model_path": "",      # 权重路径
        "num_classes": 2,
        "phi": "b2",
        "input_shape": [256, 256],
        "cuda": True,
    }

    def __init__(self, phi, path, **kwargs):
        """
        参数结构保持和原来的 SegFormer_Segmentation 一致：
        SegFormer_Segmentation_GA("b2", "./logs/best_epoch_weights.pth")
        """
        # 先写入默认配置
        self.__dict__.update(self._defaults)
        # 再覆盖用户传入的配置
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 覆盖 phi 和 model_path
        if phi != "":
            self.phi = phi
        if path != "":
            self.model_path = path

        # 二分类调色板
        if self.num_classes == 2:
            self.colors = [(0, 0, 0), (255, 255, 255)]

        self.generate()

    def generate(self, onnx=False):
        """
        实例化 SegFormer_GA, 加载权重, 搬到 GPU
        """
        # dual=True 表示使用双分支(RGB + Noise)
        self.net = SegFormer_GA(
            num_classes=self.num_classes,
            phi=self.phi,
            dual=True,
            pretrained=False
        )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 训练时保存的是：
        # state = {"epoch": ..., "state_dict": model.state_dict(), "optimizer": ...}
        # 所以这里取 ['state_dict']
        state_dict = torch.load(self.model_path, map_location=device)['state_dict']
        self.net.load_state_dict(state_dict, strict=False)

        self.net = self.net.eval()

        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image_resize(self, image):
        image = cvtColor(image)

        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # resize 到网络输入大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # --------- 关键改动从这里开始 ---------
            outputs = self.net(images, True)   # 可能是 Tensor 或 (seg_logits, cls_logits)

            # 统一拆开
            if isinstance(outputs, tuple):
                seg_logits, cls_logits = outputs       # seg_logits: [B,C,H,W]
            else:
                seg_logits, cls_logits = outputs, None # 兼容老版只返回 seg 的情况

            # 取第一个样本：seg_logit: [C,H,W]
            seg_logits = seg_logits[0]

            # 如果你想要图像级概率，这里也可以算出来先留着：
            if cls_logits is not None:
                # cls_logits: [B, num_img_classes]
                cls_prob = torch.softmax(cls_logits, dim=-1)[0].detach().cpu().numpy()
                # 例如：cls_prob[1] 就是“篡改”概率，你之后可以返回/打印/记录
            else:
                cls_prob = None
            # --------- 关键改动到这里结束 ---------

            # softmax 得到每类像素概率 [H,W,C]
            pr = F.softmax(seg_logits.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 把 padding 的区域裁掉，恢复到 nh x nw
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]

            # 再插值回原图大小
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # 取第 1 类（前景）的概率作为灰度图 mask
            seg_pred = pr[:, :, 1]
            seg_pred_new = seg_pred.reshape(orininal_h, orininal_w)
            seg_pred_new = Image.fromarray(np.uint8(seg_pred_new * 255))

            # 再做一次 argmax 得离散 mask（0/1）
            pr_cls = pr.argmax(axis=-1)

        # 着色 mask 图
        seg_img = np.reshape(
            np.array(self.colors, np.uint8)[np.reshape(pr_cls, [-1])],
            [orininal_h, orininal_w, -1]
        )
        image = Image.fromarray(np.uint8(seg_img))

        # 目前先保持原接口不变：返回 (彩色分割图, 灰度概率图)
        # 如果以后想用图像级分类结果，可以改成 return image, seg_pred_new, cls_prob
        return image, seg_pred_new, cls_prob

    def detect_image_noresize(self, image):
        """
        原图尺寸直接送入网络(不 padding / resize), 你原来的逻辑有一个 old_img 没用到, 
        我帮你去掉了。
        """
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)),
            0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images, True)
            if isinstance(outputs, tuple):
                seg_logits, cls_logits = outputs
            else:
                seg_logits, cls_logits = outputs, None

            seg_logits = seg_logits[0]

            if cls_logits is not None:
                cls_prob = torch.softmax(cls_logits, dim=-1)[0].detach().cpu().numpy()
            else:
                cls_prob = None

            pr = F.softmax(seg_logits.permute(1, 2, 0), dim=-1).cpu().numpy()
            seg_pred = pr[:, :, 1]
            seg_pred_new = seg_pred.reshape(orininal_h, orininal_w)
            seg_pred_new = Image.fromarray(np.uint8(seg_pred_new * 255))

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return image, seg_pred_new, cls_prob

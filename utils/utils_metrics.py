import os
from os.path import join
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, fbeta_score


def f_score_(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# def f_score(inputs, target, beta=1.0, smooth=1e-5, threhold=0.5,
#             fg_channel=1):
#     """
#     fg_channel: 指定哪一个通道是「篡改前景」。
#     如果以后你改了 one-hot 顺序, 只要改这个参数即可。
#     """
#     n, c, h, w = inputs.size()
#     nt, ht, wt, ct = target.size()

#     if h != ht or w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt),
#                                mode="bilinear", align_corners=True)

#     assert c == ct, f"Channel mismatch: inputs C={c}, target C={ct}"

#     prob = torch.softmax(inputs, dim=1)  # [N,C,H,W]

#     prob_fg = prob[:, fg_channel, :, :]          # [N,H,W]
#     tgt_fg  = target[..., fg_channel]            # [N,H,W]
#     tgt_fg  = (tgt_fg > 0.5).float()

#     y_score = prob_fg.detach().reshape(-1).cpu().numpy()
#     y_true  = (tgt_fg.detach().reshape(-1) > 0.5).cpu().numpy().astype(int)
#     y_pred  = (y_score > threhold).astype(int)

#     if (y_true.max() == 0) and (y_pred.max() == 0):
#         score = 1.0
#     else:
#         score = f1_score(
#             y_true,
#             y_pred,
#             average="binary",
#             zero_division=1,
#         )

#     return torch.tensor(score, dtype=torch.float32, device=inputs.device)

def f_score(
    inputs: torch.Tensor,
    masks: torch.Tensor,
    beta: float = 1.0,
    smooth: float = 1e-5,
    threhold: float = 0.5,
    fg_index: int = 1,
):
    """
    使用 sklearn.metrics.f1_score 计算最普通的像素级 F1(二分类): 

    - 网络输出 inputs: [N, C, H, W], C 至少包含 [背景, 篡改] 两类；
    - GT 使用 masks 而不是 one-hot labels: 
        * masks: [N,H,W] 或 [N,1,H,W]
        * 约定: 像素值 > 0 表示「篡改前景」, ==0 表示「背景」
    - 前景概率取自 inputs 的第 fg_index 通道(默认 1 = 篡改类)
    - 在 N*H*W 所有像素上展开成一维再用 sklearn 计算 F1

    特殊约定: 
    - 如果 GT 和 Pred 都没有前景像素 => 视为 perfect, 返回 F1 = 1.0
    """

    # ------------- 1. 取出 shapes -------------
    n, c, h, w = inputs.size()

    # masks: [N,H,W] or [N,1,H,W]
    if masks.ndim == 4:
        # [N,1,H,W] -> [N,H,W]
        if masks.size(1) != 1:
            raise ValueError(f"Expect masks with channel=1, got {masks.size()}")
        masks_2d = masks[:, 0, :, :]
    elif masks.ndim == 3:
        masks_2d = masks
    else:
        raise ValueError(f"Unsupported masks shape: {masks.size()}")

    _, mh, mw = masks_2d.size()

    # ------------- 2. 尺寸对齐 -------------
    if h != mh or w != mw:
        inputs = F.interpolate(
            inputs, size=(mh, mw), mode="bilinear", align_corners=True
        )
        h, w = mh, mw

    # ------------- 3. 前景概率(来自 logits)-------------
    # softmax 后取指定通道作为「篡改前景」概率
    prob = torch.softmax(inputs, dim=1)  # [N,C,H,W]
    if not (0 <= fg_index < prob.size(1)):
        raise ValueError(f"fg_index={fg_index} out of range for C={prob.size(1)}")

    prob_fg = prob[:, fg_index, :, :]  # [N,H,W]

    # ------------- 4. GT 前景掩膜(二值)-------------
    # 约定: mask > 0 即为前景(可能是 1/2/3, 对应不同篡改类型)
    tgt_fg = (masks_2d > 0).float()    # [N,H,W]

    # ------------- 5. 展平 + numpy -------------
    y_score = prob_fg.detach().cpu().numpy().reshape(-1)         # 概率
    y_true  = tgt_fg.detach().cpu().numpy().reshape(-1) > 0.5    # bool
    y_true  = y_true.astype(int)                                 # 0/1
    y_pred  = (y_score > threhold).astype(int)                   # 0/1

    # ------------- 6. 特殊情况: GT & Pred 都没有前景 -------------
    if (y_true.max() == 0) and (y_pred.max() == 0):
        score = 1.0
    else:
        score = f1_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=1,  # 避免除零, 同时保持“全负且预测全负=1.0”的语义
        )

    return torch.tensor(score, dtype=torch.float32, device=inputs.device)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    
    gt_imgs=[]
    for name in png_name_list:
        path_gt=os.path.join(gt_dir, name + ".png")
        if os.path.isfile(path_gt):
            mask_path = os.path.join(gt_dir, name + ".png")
        else:
            mask_path = os.path.join(gt_dir, name + ".tif")
        gt_imgs.append(mask_path)
            
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))  

        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('=====>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    print('=====> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int_), IoUs, PA_Recall, Precision
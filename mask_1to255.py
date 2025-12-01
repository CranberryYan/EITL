from PIL import Image
import numpy as np
import os

mask_path = r"/mnt/e/jwj/test_data/Mask/1167.png"

# 读取原始 0/1 mask
img = Image.open(mask_path)
arr = np.array(img)

print("原图信息：")
print(" mode:", img.mode)
print(" shape:", arr.shape)
print(" dtype:", arr.dtype)
print(" min:", arr.min(), "max:", arr.max())
print(" unique:", np.unique(arr))

# 映射到 0/255
arr_255 = (arr * 255).astype(np.uint8)

print("\n映射到 0/255 之后：")
print(" shape:", arr_255.shape)
print(" dtype:", arr_255.dtype)
print(" min:", arr_255.min(), "max:", arr_255.max())
print(" unique:", np.unique(arr_255))

# 保存新图
img_255 = Image.fromarray(arr_255, mode="L")

save_dir = './'
save_path = os.path.join(save_dir, "1167_255.png")
img_255.save(save_path)

print("\n新图已保存为：", save_path)

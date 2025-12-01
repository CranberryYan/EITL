import os
import shutil

# 配置路径(请根据实际情况修改)
small_folder = '/media/yst/Elements SE/jwj/my_data/SegmentationClass'      # 小文件夹, 参考顺序和文件名
large_folder = '/media/yst/Elements SE/jwj/EITL_datasets/SegmentationClass0255'      # 大文件夹, 所有图片来源
output_folder = '/media/yst/Elements SE/jwj/my_data/SegmentationClass0255'    # 输出新文件夹

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取小文件夹中的图片文件(保持顺序)
image_names = os.listdir(small_folder)
image_names.sort()  # 如果你想保持系统顺序, 可以去掉这行

# 遍历并按顺序复制图片
copied = 0
missing = []

for img_name in image_names:
    src_path = os.path.join(large_folder, img_name)
    dst_path = os.path.join(output_folder, img_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied += 1
    else:
        missing.append(img_name)

print(f"\n✅ 已复制 {copied} 张图片到：{output_folder}")
if missing:
    print(f"⚠️ 有 {len(missing)} 张图片未找到：")
    for name in missing[:10]:  # 仅显示前10条
        print(f"  - {name}")
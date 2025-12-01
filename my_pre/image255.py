import os
import shutil

# 路径配置
small_folder = '/media/yst/Elements SE/jwj/my_data/SegmentationClass'        # 小文件夹, 包含目标图片名
large_folder = '/media/yst/Elements SE/jwj/EITL_datasets/SegmentationClass0255'        # 大文件夹, 从这里抽取图片
output_folder = 'media/yst/Elements SE/jwj/my_data/SegmentationClass0255'      # 输出新文件夹

# 创建输出文件夹(如果不存在)
os.makedirs(output_folder, exist_ok=True)

# 获取小文件夹中所有文件名(保持顺序)
image_names = sorted(os.listdir(small_folder))  # 如有特殊顺序要求, 可改为不排序

# 遍历复制
for name in image_names:
    src_path = os.path.join(large_folder, name)
    dst_path = os.path.join(output_folder, name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"[警告] 未找到文件：{src_path}")

print(f"✅ 共处理 {len(image_names)} 张图片, 已复制到 {output_folder}")
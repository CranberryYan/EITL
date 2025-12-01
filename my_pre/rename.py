import os

# 设置你的图片文件夹路径
folder_path = '/media/yst/Elements SE/jwj/test_data/Mask0255'  # ← 修改为你的实际路径

# 遍历文件夹中所有 .png 文件, 按文件名排序
files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

for file in files:
    old_path = os.path.join(folder_path, file)
    if '_gt.png' in file:
        print(f"已跳过已修改文件：{file}")
        continue
    name, ext = os.path.splitext(file)
    new_name = f"{name}_gt{ext}"
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"重命名：{file} → {new_name}")

print("✅ 所有图片重命名完成！")

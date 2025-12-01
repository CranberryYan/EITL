import os

def count_images_in_folder(folder_path):
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')
    
    # 统计该文件夹中符合扩展名的图片数量
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            count += 1
    return count

# 设置两个文件夹路径(请替换为你自己的路径)
folder1 = '/media/yst/Elements SE/jwj/EITL_datasets/JPEGImages'
folder2 = '/media/yst/Elements SE/jwj/EITL_datasets/SegmentationClass'
folder3 = '/media/yst/Elements SE/jwj/EITL_datasets/SegmentationClass0255'

# 分别统计数量
count1 = count_images_in_folder(folder1)
count2 = count_images_in_folder(folder2)
count3 = count_images_in_folder(folder3)

print(f"文件夹1 ({folder1}) 中有 {count1} 张图片。")
print(f"文件夹2 ({folder2}) 中有 {count2} 张图片。")
print(f"文件夹3 ({folder3}) 中有 {count3} 张图片。")
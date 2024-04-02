import os
import shutil
from tqdm import tqdm

def move_subdirectories(directory, destination):
    for root, dirs, files in tqdm(os.walk(directory)):
        if not dirs:  # 如果当前目录没有子目录
            jpg_files = [f for f in files if f.endswith('.jpg')]
            if len(jpg_files) < 9:
                parent_directory = os.path.basename(os.path.dirname(root))
                subdirectory = os.path.basename(root)  # 获取最子目录的名称
                destination_path = os.path.join(destination, parent_directory, subdirectory)  # 构建目标路径
                shutil.move(root, destination_path)  # 移动最子目录到目标路径


# 指定当前目录
current_directory = "/mnt/sda/cxh/data/training_data_youtube/youtube_crop_face/"
destination = "/mnt/sda/cxh/data/training_data_youtube/less9"
move_subdirectories(current_directory, destination)
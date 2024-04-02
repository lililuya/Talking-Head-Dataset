import os


import os

def print_directories_with_few_subdirectories(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        dir_len = len(os.listdir(item_path))
        if dir_len<5:
            print(item_path)
# 示例用法
directory = "/mnt/sda/cxh/data/training_data_youtube/youtube_crop_face"

print_directories_with_few_subdirectories(directory)
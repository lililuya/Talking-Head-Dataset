import glob
import json
import os
import subprocess
from tqdm import tqdm
import cv2
import numpy as np
import concurrent.futures
import torch
import csv
import facer
from PIL import Image

import multiprocessing



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cuda:7" 
# device = "cuda:6" 
# device = "cuda:5"  
# device = "cuda:4"
# image: 1 x 3 x h x w
# out_csv_dir="/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_csv"
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_aligner = facer.face_aligner('farl/ibug300w/448', device=device) # optional: "farl/wflw/448", "farl/aflw19/448"


def read_frame(path: str) -> torch.Tensor:
    """Read an image from a given path.

    Args:
        path (str): The given path.
    """
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))  # W H C
    return torch.from_numpy(np_image)

def read_resize_frame(path, target_width, target_height):
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))  # H W C 
    original_height, original_width = np_image.shape[0], np_image.shape[1]
    ratio_w = original_width / target_width
    ratio_h = original_height / target_height
    # print(ratio_w)
    ratio = (ratio_w, ratio_h)
    img_resize = cv2.resize(np_image, (target_width, target_height))
    return torch.from_numpy(img_resize),  ratio

# 根据frame得到关键点
def get_lmk_csv_frame(frame_dir, out_csv_dir):
    basename = os.path.basename(frame_dir)
    csv_path = os.path.join(out_csv_dir, basename + "_lmk.csv")
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        column_names = ['frame_index'] + ['x_{}'.format(i) for i in range(68)] + ['y_{}'.format(i) for i in range(68)]
        writer.writerow(column_names)
        # print(frame_dir)
        for frame_index, frame in enumerate(sorted(os.listdir(frame_dir))):
            frame_path = os.path.join(frame_dir, frame)
            # 读取视频帧
            frame_np, ratio = read_resize_frame(frame_path, target_width=640, target_height=360)
            image = facer.hwc2bchw(frame_np).to(device=device)  # filepath
            with torch.inference_mode():
                faces = face_detector(image)
            with torch.inference_mode():
                faces = face_aligner(image, faces)
                lmk = faces['alignment'].cpu().numpy()
                landmarks = lmk[0]
                # 调整至原图大小
                adjusted_landmarks = []
                for landmark in landmarks:
                    x, y = landmark
                    adjusted_x = x * ratio[0]
                    adjusted_y = y * ratio[1]
                    # print(adjusted_x)
                    adjusted_landmarks.append([adjusted_x, adjusted_y])
                adjusted_landmarks = np.array(adjusted_landmarks)
                rounded_landmarks = [[round(coord, 1) for coord in landmark] for landmark in adjusted_landmarks]
                x_coord = [coord[0] for landmark in rounded_landmarks for coord in landmark]
                y_coord = [coord[1] for landmark in rounded_landmarks for coord in landmark]
                writer.writerow([frame_index]+ [x_coord]+ [y_coord])

if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    # root_dir = "/mnt/sda/cxh/data/test/Youtubev1-000.mp4"
    frame_dir = "/mnt/sda/cxh/data/training_data_youtube/youtube/"
    
    # get_lmk_csv_video(root_dir, out_csv_dir)
    # 将目录分为三份
    frame_dir_path = sorted(os.listdir(frame_dir))
    # print(frame_dir_path)
    whole_frame_path_list = []
    for item in frame_dir_path:
        whole_frame_path = os.path.join(frame_dir, item)
        whole_frame_path_list.append(whole_frame_path)
    
    # path = whole_frame_path_list[0:100]
    out_csv_dir = "/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_csv0-100"

    # path = whole_frame_path_list[100:200]
    # out_csv_dir = "/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_csv100-200"

    path = whole_frame_path_list[0:100]
    # out_csv_dir = "/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_csv200-300"
    # out_csv_dir = "/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_farl"s


    # for item in sorted(path):
    #     get_lmk_csv_frame(item, out_csv_dir)
    # path = whole_frame_path_list[300:]
    # out_csv_dir = "/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_csv300-400"

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_lmk_csv_frame, item, out_csv_dir) for item in path]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occur:{e}")

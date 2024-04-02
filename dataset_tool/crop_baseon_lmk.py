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

face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_aligner = facer.face_aligner('farl/ibug300w/448', device=device) # optional: "farl/wflw/448", "farl/aflw19/448"

def read_frame(path: str) -> torch.Tensor:
    """Read an image from a given path.

    Args:
        path (str): The given path.
    """
    image = Image.open(path)
    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# image: 1 x 3 x h x w

def get_lmk_csv_video(vfile, out_csv_dir):
    video = cv2.VideoCapture(vfile)
    basename = os.path.splitext(os.path.basename(vfile))[0]
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    csv_path = os.path.join(out_csv_dir, basename + "_lmk.csv")
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        column_names = ['frame_index'] + ['x_{}'.format(i) for i in range(68)] + ['y_{}'.format(i) for i in range(68)]
        writer.writerow(column_names)
        # 读取视频帧
        for frame_index in tqdm(range(frame_count), basename):
            ret, frame = video.read()  # H W C-->BGR
            if ret is None:
                print("None frame detect!!!",frame_index)
                continue
            image = facer.hwc2bchw(read_frame(frame)).to(device=device)  # filepath
            with torch.inference_mode():
                faces = face_detector(image)
            with torch.inference_mode():
                faces = face_aligner(image, faces)
                lmk = faces['alignment'].cpu().numpy()

                # 写入人脸关键点数据
                landmarks = lmk[0]
                rounded_landmarks = [[round(coord, 1) for coord in landmark] for landmark in landmarks]
                writer.writerow([frame_index] + [coord for landmark in rounded_landmarks for coord in landmark])


# 根据frame得到关键点
def get_lmk_csv_frame(frame_dir, gpu_id):
    # 设置当前进程应使用的GPU设备
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    basename = os.path.basename(frame_dir)
    csv_path = os.path.join(out_csv_dir, basename + "_lmk.csv")
    
    # basename = os.path.splitext(os.path.basename(frame_path))[0]
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        column_names = ['frame_index'] + ['x_{}'.format(i) for i in range(68)] + ['y_{}'.format(i) for i in range(68)]
        writer.writerow(column_names)
        # print(frame_dir)
        for frame_index, frame in enumerate(sorted(os.listdir(frame_dir))):
            # print(frame_path)
            frame_path = os.path.join(frame_dir, frame)
            print(frame_path)
            # 读取视频帧
            image = facer.hwc2bchw(read_frame(frame_path)).to(device=device)  # filepath
            with torch.inference_mode():
                faces = face_detector(image)
            with torch.inference_mode():
                faces = face_aligner(image, faces)
                lmk = faces['alignment'].cpu().numpy()
                landmarks = lmk[0]
                rounded_landmarks = [[round(coord, 1) for coord in landmark] for landmark in landmarks]
                writer.writerow([frame_index] + [coord for landmark in rounded_landmarks for coord in landmark])


def allocate_processes_to_gpus(folder_list, processes_per_gpu=3):
    gpu_count = torch.cuda.device_count()
    all_tasks = []

    # 为每个gpu创建任务列表
    for gpu_id in range(gpu_count):
        for _ in range(processes_per_gpu):
            # 这里分配folder_list中的任务
            tasks_for_gpu = folder_list[gpu_id::gpu_count]
            for folder in tasks_for_gpu:
                all_tasks.append((folder,gpu_id))
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=processes_per_gpu * gpu_count) as executor:
    #     futures = []
    #     for i, frame_dir in enumerate(folder_list):
    #         gpu_id = i % gpu_count  # 确保均匀地分配任务到每个GPU
    #         futures.append(executor.submit(get_lmk_csv_frame, frame_dir, gpu_id))
        
        # 等待所有任务完成
        # for future in concurrent.futures.as_completed(futures):
        #     try:
        #         future.result()
        #     except Exception as e:
        #         print(f"Exception occurred: {e}")
        # 使用ProcessPoolExecutor执行所有任务
                

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     future_to_task = {executor.submit(get_lmk_csv_frame, folder, gpu_id): (folder, gpu_id) for folder, gpu_id in all_tasks}
    #     for future in concurrent.futures.as_completed(future_to_task):
    #         folder, gpu_id = future_to_task[future]
    #         try:
    #             # 获取结果，如果有异常会在这里抛出
    #             future.result()
    #             print(f"Task completed: {folder} on GPU {gpu_id}")
    #         except Exception as e:
    #             print(f"Exception occurred in task {folder} on GPU {gpu_id}: {e}")



if __name__=="__main__":
    # # root_dir = "/mnt/sda/cxh/data/test/Youtubev1-000.mp4"
    # frame_dir = "/mnt/sda/cxh/data/training_data_youtube/youtube/600_clip004"
    # out_csv_dir = "/mnt/sdb/cxh/liwen/DINet_optimized/tools/lmk_csv"
    # # get_lmk_csv_video(root_dir, out_csv_dir)
    # get_lmk_csv_frame(frame_dir, out_csv_dir)
    multiprocessing.set_start_method('spawn', True)  # 用于兼容CUDA
    dir = "/mnt/sda/cxh/data/training_data_youtube/youtube"
    whole_path_list = []
    for frame in sorted(os.listdir(dir)):
        whole_path = os.path.join(dir, frame)
        whole_path_list.append(whole_path)
    allocate_processes_to_gpus(whole_path_list)
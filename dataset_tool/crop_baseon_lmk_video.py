import glob
import json
import os
import subprocess
from tqdm import tqdm
import cv2
import numpy as np
import csv
from config.config import DataProcessingOptions
from utils.data_processing import compute_crop_radius, load_landmark_openface
from utils.deep_speech import DeepSpeech
import torch
import facer
from PIL import Image
import random
import concurrent.futures
import shutil

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda:7"

face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_aligner = facer.face_aligner('farl/ibug300w/448', device=device) # optional: "farl/wflw/448", "farl/aflw19/448"

json_file = "/mnt/sdb/cxh/liwen/DINet_optimized/video_list/mask_face_undetected.json"
ill_data = {}

# 没有分9帧的操作
def compute_crop_radius(video_size, landmark_data_clip, random_scale=None):
    """
    judge if crop face and compute crop radius
    """
    video_w, video_h = video_size[0], video_size[1]
    landmark_max_clip = np.max(landmark_data_clip, axis=1)
    # scale_w = video_w/512
    # scale_h = video_h/512
    if random_scale is None:
        random_scale = random.random() / 10 + 1.05
    else:
        random_scale = random_scale
    # radius_h是特征点29（鼻尖）到人脸最高点的垂直距离
    # radius_w是特征点48（嘴角一侧）到54（嘴角另一侧）的水平距离
    radius_h = (landmark_max_clip[:, 1] - landmark_data_clip[:, 29, 1]) * random_scale
    radius_w = (
        landmark_data_clip[:, 54, 0] - landmark_data_clip[:, 48, 0]
    ) * random_scale
    radius_clip = np.max(np.stack([radius_h, radius_w], 1), 1) // 2
    radius_max = np.max(radius_clip)
    radius_max = (np.int32(radius_max / 4) + 1) * 4
    radius_max_1_4 = radius_max // 4
    clip_min_h = landmark_data_clip[:, 29, 1] - radius_max   # (1,68,2)  # 鼻尖垂直中心
    clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 2 + radius_max_1_4 # 
    clip_min_w = landmark_data_clip[:, 33, 0] - radius_max - radius_max_1_4 # 鼻子的下部中心，人脸的水平中心
    clip_max_w = landmark_data_clip[:, 33, 0] + radius_max + radius_max_1_4
    # 检测是否超过了视频的边界
    if min(clip_min_h.tolist() + clip_min_w.tolist()) < 0:
        return False, None
    elif max(clip_max_h.tolist()) > video_h:
        return False, None
    elif max(clip_max_w.tolist()) > video_w:
        return False, None
    elif max(radius_clip) > min(radius_clip) * 1.5:
        return False, None
    else:
        return True, radius_max

def read_resize_frame(img_np, target_width, target_height):
    np_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # H W C 
    original_height, original_width = np_image.shape[0], np_image.shape[1]
    ratio_w = original_width / target_width
    ratio_h = original_height / target_height
    ratio = (ratio_w, ratio_h)
    img_resize = cv2.resize(np_image, (target_width, target_height))
    return torch.from_numpy(img_resize),  ratio


def get_lmk_csv_video(vfile, img_out_dir, out_csv_dir, clip_length=9):
    video = cv2.VideoCapture(vfile)
    basename = os.path.splitext(os.path.basename(vfile))[0]
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    csv_path = os.path.join(out_csv_dir, basename + "_lmk.csv")
    img_out_path = os.path.join(img_out_dir, basename)
    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        column_names = ['frame_index'] + ['x_{}'.format(i) for i in range(68)] + ['y_{}'.format(i) for i in range(68)]
        writer.writerow(column_names)
        for frame_index in tqdm(range(frame_count), basename):
            ret, frame = video.read()  # H W C-->BGR
            video_h, video_w = frame.shape[0], frame.shape[1]
            if ret is None:
                print("None frame detect!!!", frame_index)
                continue
            frame_np, ratio = read_resize_frame(frame, target_width=640, target_height=360)
            image = facer.hwc2bchw(frame_np).to(device=device)  # filepath
            with torch.inference_mode():
                faces = face_detector(image)
                # print(faces)
                if 'image_ids' not in faces.keys():
                    try:  # 检测不到就执行下面操作
                        os.remove(csv_path)
                        shutil.rmtree(img_out_path) 
                        ill_data[vfile] = frame_index
                        with open(json_file,"a") as file:
                            json.dump(ill_data, file)
                    except:
                        print("delete failed!")
                        exit()
                    break
            with torch.inference_mode():
                faces = face_aligner(image, faces)
                lmk = faces['alignment'].cpu().numpy()
                landmarks = lmk[0]
                # Todo2: 对landmark还原
                adjusted_landmarks = []
                for landmark in landmarks:
                    x, y = landmark
                    adjusted_x = x * ratio[0]
                    adjusted_y = y * ratio[1]
                    adjusted_landmarks.append([adjusted_x, adjusted_y])
                adjusted_landmarks = np.array(adjusted_landmarks)
                # Todo3: 保存需要的landmark
                rounded_landmarks = [[round(coord, 1) for coord in landmark] for landmark in adjusted_landmarks]
                lmk_coord = [coord for landmark in rounded_landmarks for coord in landmark]
                writer.writerow([frame_index]+ lmk_coord)
                
                # 处理landmark
                coordinates = np.array([(lmk_coord[i], lmk_coord[i+1]) for i in range(0, len(lmk_coord), 2)])
                landmark_data = coordinates.astype(np.int32) # (1,68,2)
                landmark_data_crop = np.reshape(landmark_data,(1,68,2)) 
                # 计算crop区域
                crop_flag, radius_clip = compute_crop_radius((video_w, video_h), landmark_data_crop)      
                if not crop_flag:
                    continue
                radius_clip_1_4 = radius_clip // 4  
                res_crop_face_frame_path = os.path.join(img_out_path, str(frame_index).zfill(6) + ".jpg")

                # print(res_crop_face_frame_path)
                if os.path.exists(res_crop_face_frame_path):
                    print("跳过已处理的帧: ", res_crop_face_frame_path)
                    continue
                
                frame_landmark = landmark_data_crop[0 , :, :]
                # 切片crop
                crop_face_data = frame[
                    frame_landmark[29, 1]
                    - radius_clip : frame_landmark[29, 1]
                    + radius_clip * 2
                    + radius_clip_1_4,
                    frame_landmark[33, 0]
                    - radius_clip
                    - radius_clip_1_4 : frame_landmark[33, 0]
                    + radius_clip
                    + radius_clip_1_4,
                    :,
                ].copy()
                cv2.imwrite(res_crop_face_frame_path, crop_face_data)

def split_list(video_list, step, output_folder):
    if  not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 计算分成的份数
    count = len(video_list) // step + 1
    for i in range(count):
        start = i*step
        end = (i+1)*step
        part_lst = video_list[start:end]
        filename = f"part_{i+1}.txt"
        file_path = os.path.join(output_folder, filename)
        with open(file_path, "w") as f:
            for item in part_lst:
                f.write(str(item) + "\n")

if __name__=="__main__":
    root_dir = "/mnt/sda/cxh/data/Youtubev1_use_clip/"
    img_out_dir = "/mnt/sda/cxh/data/training_data_youtube/youtube_crop_face"
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    out_csv_dir = "/mnt/sda/cxh/data/training_data_youtube/youtube_csv/"
    if not os.path.exists(out_csv_dir):
        os.makedirs(out_csv_dir)
    
    mp4_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                mp4_files.append(file_path)
    mp4_files = sorted(mp4_files)

    # split_list(mp4_files, step=400, output_folder="/mnt/sdb/cxh/liwen/DINet_optimized/video_list")
    
    # mp4_list_path = "/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_1.txt" #  [1] 1422551
    # mp4_list_path = "/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_2.txt"   #  [2] 1422968
    # mp4_list_path = "/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_3.txt" #  [3] 1423276
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_4.txt"  #  [4] 1423577
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_5.txt"  #  [5] 1423881
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_6.txt"    # [6] 1424099


    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_7.txt"  #[7] 1424473
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_8.txt"  # [8] 1424766
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_9.txt"  # [9] 1424986
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_10.txt" # [10] 1425266
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_11.txt" # [11] 1425560
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_12.txt" # [12] 1425898

    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_13.txt"  # [13] 1426224
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_14.txt"  # [14] 1426530
    # mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_15.txt"  # [15] 1426827
    mp4_list_path ="/mnt/sdb/cxh/liwen/DINet_optimized/video_list/part_16.txt"    # [16] 1427132

    mp4_list = []
    # mp4_list_path = "/mnt/sdb/cxh/liwen/DINet_optimized/video_list/test.txt"
    
    with open(mp4_list_path, "r") as file:
        for line in file:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            mp4_list.append(line)
    for vfile in mp4_list:
        get_lmk_csv_video(vfile, img_out_dir, out_csv_dir)
    # # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #     futures = [executor.submit(get_lmk_csv_video, vfile, img_out_dir, out_csv_dir) for vfile in vfile_list]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"An error occur:{e}")
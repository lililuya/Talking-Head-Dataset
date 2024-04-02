import os
import cv2
import json
from tqdm import tqdm
import numpy as np
import face_detection
import argparse
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
from insightface_func.face_detect_crop_multi import Face_detect_crop
from glob import glob
import argparse
"""
处理视频，记录视频无人脸的部分
"""

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=4, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=16, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset",  default="/mnt/sda/cxh/data/test/",required=False)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default="/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/json" ,required=False)
parser.add_argument("--out_json_path", help="Root folder of the preprocessed dataset", default="/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/json" ,required=False)

args = parser.parse_args()
# json_file = "/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/json/test.json"
out_json_path = "/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/json_50/"
# def process_video_segment(vfile, start_frame, end_frame, gpu_id, progress, batch_size=3):
#     video_stream = cv2.VideoCapture(vfile)
#     video_stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(gpu_id))
#     frames_with_none_face = []
#     for frame_index in range(start_frame, end_frame, batch_size):
#         frames = []
#         for _ in range(batch_size):
#             ret, frame = video_stream.read()
#             if not ret:
#                 break
#             frames.append(frame)
#         preds = fa.get_detections_for_batch(np.asarray(frames))
#         for j, f in enumerate(preds):
#             if f is None:
#                 frames_with_none_face.append(start_frame + frame_index + j)
#         progress.update(len(frames))
#     return frames_with_none_face


# def process_video_file(vfile, args, progress):
#     video_stream = cv2.VideoCapture(vfile)
#     total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#     segment_size = total_frames // args.ngpu  # 总
#     segments = []
#     start_frame = 0
#     for i in range(args.ngpu - 1):
#         end_frame = start_frame + segment_size
#         segments.append((start_frame, end_frame))
#         start_frame = end_frame
#     segments.append((start_frame, total_frames))
#     frames_with_none_face = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=args.ngpu) as executor:
#         futures = []
#         for i, (start_frame, end_frame) in enumerate(segments):
#             future = executor.submit(process_video_segment, vfile, start_frame, end_frame, i, progress, args.batch_size)
#             futures.append(future)
#         for future in concurrent.futures.as_completed(futures):
#             segment_result = future.result()
#             frames_with_none_face.extend(segment_result)
#     return frames_with_none_face


def initialize_model():
     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
     detect_model = Face_detect_crop(name='antelope', root='./insightface_func/models')
     detect_model.prepare(ctx_id = 0, det_thresh=0.6, det_size=(640,640), mode = None, crop_size=384, ratio=0.8)
     return detect_model

def process_video_face(video_paths_list, max_workers=10):
     models = []
     for _ in range(max_workers):
        model = initialize_model()
        models.append(model)

     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, video_path in enumerate(video_paths_list):
            # print(video_path)
            model = models[i % max_workers]
            future = executor.submit(single_detect, model, video_path)
            futures.append(future)
        
        wait(futures, return_when=ALL_COMPLETED)

        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            

def get_path_list(root_dir="/mnt/sda/cxh/data/Youtubev1_25fps_0_300"):
      file_paths_list = []
      for file in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file)
            file_paths_list.append(file_path)
      return file_paths_list


def single_detect(detect_model, vfile, frame_interv=10):
      frame_with_no_face = []
      basepart = os.path.splitext(vfile)[0]

      basename = basepart.split("/")[-1]
      json_file = os.path.join(out_json_path, basename+".json")

      video = cv2.VideoCapture(vfile)
      frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

      for frame_index in tqdm(range(frame_count), "basename:"):
            if frame_index % frame_interv ==0:
                  ret, frame = video.read()
                  if ret:
                        detect_results = detect_model.get_boxes(frame)
                        if detect_results is None:
                              frame_with_no_face.append(frame_index)
      with open(json_file,"w") as file:
            json.dump(frame_with_no_face, file)



if __name__ == '__main__':
      video_paths_list = get_path_list() 
      video_paths_list = video_paths_list # [1] 2025031
      process_video_face(video_paths_list)

      # video_paths_list = get_path_list() 
      # video_paths_list = video_paths_list[300:-1]  # [2] 2026400
      # process_video_face(video_paths_list)
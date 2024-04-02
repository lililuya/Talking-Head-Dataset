import json
import subprocess
import concurrent.futures
from moviepy.editor import VideoFileClip, concatenate_videoclips
import json
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import math

"""
根据json处理视频
"""
video_fps = 25
json_path = "/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/json/336.json"
max_segment_duration = 60 # 1min的最大
min_segment_duration = 20 # 最小的阈值

# 按10为间隔分组
def group_json():
    with open(json_file, 'r') as file:
        data = json.load(file)
    grouped_data = []
    current_group = []

    for item in data:
        if not current_group or item - current_group[-1] == 10:
            current_group.append(item)
        else:
            grouped_data.append(current_group)
            current_group = [item]

    if current_group:
        grouped_data.append(current_group)
    for group in grouped_data:
        print(group)

def process_video(params):
    video_path, json_path, output_path, fps = params
    cut_video(video_path, json_path, output_path, fps)

def group_indices(no_face_indices, frame_interval=10):
    groups = []
    # print(no_face_indices)
    current_group = [no_face_indices[0]]
    
    for index in no_face_indices[1:]:
        if index - current_group[-1] == frame_interval:
            current_group.append(index)
        else:
            groups.append(current_group)
            current_group = [index]
    groups.append(current_group)  
    # print(groups)
    return groups

def cut_video(video_path, json_path, output_dir, fps):
    with open(json_path, 'r') as f:
        no_face_indices = json.load(f)

    if not no_face_indices:
        print("No unface frame",json_path)
        return

    video_basename = os.path.basename(os.path.splitext(video_path))
    output_path = os.path.join(output_dir, video_basename)
    
    video = VideoFileClip(video_path)
    groups = group_indices(no_face_indices)
    
    cut_segments = []
    for group in groups:
        start_frame, end_frame = group[0], group[-1]
        start_time = max(0, start_frame / fps - 5)
        end_time = min(video.duration, end_frame / fps + 5) 
        cut_segments.append((start_time, end_time))
    

    keep_segments = []
    last_end = 0
    
    # 从0到开始无人脸的帧的前5s保存，依次类推
    for start, end in cut_segments:
        keep_segments.append((last_end, start))
        last_end = end
    keep_segments.append((last_end, video.duration))  
    final_clips = [video.subclip(start, end) for start, end in keep_segments if start < end]
    
    out_clip_index = 0
    for clip in final_clips:
        # 获取片段的持续时间
        duration = video.duration
        if duration < 20:
            # print("less than 20s, throw")
            continue
        elif duration > 60:
            num_segments = math.ceil(duration / max_segment_duration)
            for i in range(num_segments):
                start_time = i*max_segment_duration
                end_time = min((i+1) * max_segment_duration, duration)
                segment = clip.subclip(start_time, end_time)
                output_file_path = output_path + f"{video_basename:03}_clip{out_clip_index:03}.mp4"
        else:
            output_file_path = f"{video_basename:03}_clip{out_clip_index:03}"
            clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac")
            out_clip_index += 1
            
    video.close()
    for clip in final_clips:
        clip.close()


if __name__=="__main__":
    # group_json()
    # with open(json_path, 'r') as f:
    #     no_face_indices = json.load(f)
    # group_indices(no_face_indices)
    json_file = "/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/json_50"
    video_dir = '/mnt/sda/cxh/data/Youtubev1_25fps_0_300'
    output_dir = '/mnt/sda/cxh/data/Youtubev1_25fps_0_300_no_face'
    video_files = [f for f in sorted(os.listdir(video_dir)) if f.endswith('.mp4')]

    tasks = []
    # for filename in tqdm(sorted(os.listdir(video_dir))):
    for filename in video_files:
        video_path = os.path.join(video_dir, filename)
        vidname = os.path.splitext(os.path.basename(filename))[0]
        json_path = os.path.join(json_file, vidname + '.json')
        output_path = os.path.join(output_dir, filename)
        
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        video_clip.close()
        tasks.append((video_path, json_path, output_path, fps))
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(process_video, tasks)  # [1] 2172335





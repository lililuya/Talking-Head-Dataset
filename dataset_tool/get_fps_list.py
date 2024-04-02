from moviepy.editor import VideoFileClip
import os
import json
from tqdm import tqdm

json_file = "/mnt/hd3/liwen/middle_process_file/youtubev1_0_300_fps.json"

def get_mp4_framerate(root_dir):
    fps ={}
    for file_path in tqdm(sorted(os.listdir(root_dir))):
        file_path = os.path.join(root_dir, file_path)
        try:
            video = VideoFileClip(file_path)
        except Exception as e:
            print("Error", file_path)
        framerate = video.fps
        fps[file_path] = framerate
    video.close()
    return fps

root_dir = '/mnt/hd/data/YouTube_25'
fps = get_mp4_framerate(root_dir)
with open(json_file, 'w') as file:
    data = json.dump(fps, file)

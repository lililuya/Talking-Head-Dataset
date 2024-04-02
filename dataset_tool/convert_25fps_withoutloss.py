import subprocess
import os
import concurrent.futures
from tqdm import tqdm


template = "ffmpeg -y -i {} -vf 'setpts=1*PTS' -r 25 -loglevel quiet {} "
input_video_path = "/mnt/hd/data/HDTF"
output_video_path = "/mnt/hd/data/HDTF5_25"
n_processes = 32
source_dir = os.listdir(input_video_path)

def convert_25fps_with_progress(name_video):
    video = os.path.join(input_video_path, name_video)
    new_video = os.path.join(output_video_path, name_video)
    command = template.format(video, new_video)
    subprocess.call(command, shell=True)
    pbar.update(1)

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(n_processes) as executor:
        inputs = [x for x in source_dir]
        total_videos = len(inputs)
        with tqdm(total=total_videos, desc="Converting videos", unit="video") as pbar:
            # 使用 convert_25fps_with_progress 代替原来的 convert_25fps
            executor.map(convert_25fps_with_progress, inputs)

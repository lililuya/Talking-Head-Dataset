import os
import subprocess
import concurrent.futures
from tqdm import tqdm


n_processes = 32
template = "ffmpeg -i {} -c copy -c:a aac {}  -strict -2  -loglevel quiet"

root_dir = "/mnt/sda/cxh/data/Youtubev1"
out_dir = "/mnt/sda/cxh/data/Youtubev1_mp4"
source_dir = os.listdir(root_dir)

def convert_mp4(mkv_file):
    mkv_path = os.path.join(root_dir, mkv_file)
    basename = os.path.splitext(mkv_file)[0]
    basename_mp4 = basename + ".mp4"
    output_template = os.path.join(out_dir, basename_mp4)
    # print(output_template)
    command = template.format(mkv_path, output_template)
    print(command)
    subprocess.call(command, shell=True)
    pbar.update(1)  # 更新进度条

def remove_mp4(root_dir):
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if item_path.endswith(".mp4"):
            os.remove(item_path)

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(n_processes) as executor:
        inputs = [x for x in sorted(source_dir)]
        total_videos = len(inputs)
        with tqdm(total=total_videos, desc="Converting videos", unit="video") as pbar:
            executor.map(convert_mp4, inputs)
    # remove_mp4(root_dir)
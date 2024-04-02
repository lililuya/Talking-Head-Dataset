import subprocess
import os
from tqdm import tqdm
import json
from multiprocessing import Pool
import argparse
from functools import partial
import concurrent.futures

parser =argparse.ArgumentParser(description="添加相关的参数")
parser.add_argument('-root_dir', default="/mnt/sda/cxh/data/Youtube_use", type=str)
parser.add_argument('-out_dir', default="/mnt/sda/cxh/data/Youtubev1_use_clip", type=str)
parser.add_argument('-json_file', default="/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/file/Youtubev1_use.json", type=str)
parser.add_argument('-duration_seed', default=60, type=int)

args = parser.parse_args()
# root_dir = "/mnt/sda/cxh/data/Youtubev1_25fps_300-1000"
# out_dir = "/mnt/sda/cxh/data/Youtubev1_clip"
# json_file = "/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/file/Youtubev1_du.json"

# duration_seed = 100
# duration_str = '00:01:40'
template = "ffmpeg -i {} -c copy -map 0 -f  segment -segment_time {}  -reset_timestamps 1 -loglevel quiet {}"
# template2 = "ffmpeg -i {} -ss {} -t {} -c copy -loglevel quiet {}"
template2 = "ffmpeg -i {} -ss {} -t {}  -loglevel quiet {}"


def get_duration(json_file):
    # 获得一个duration的列表
    with open(json_file, 'r') as f:
        data = json.load(f)
        return data

# 按目录处理，比较低效
def clip_vid2_youtube(root_dir, out_dir):
    count = 0
    data = get_duration(args.json_file)
    for vid_file in tqdm(os.listdir(root_dir),"progress:"):
        # basename = os.path.basename(vid_file)[:11]  # 现在基本名字变了
        basename = os.path.splitext(vid_file)
        vid_path = os.path.join(root_dir, vid_file)
        # 获取duration, 通过读取已经获取到的字典
        duration = data[vid_path] # 获取指定路径的键值
        out_path = os.path.join(out_dir, basename)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if duration <= args.duration_seed:  # 第一种情况是总时长小于持续的时间时，这个地方应该不存在，但是为了检验保留一步
            print("less than duration_seed")
            count += 1
            out_temp= os.path.join(out_path, f"{basename}_clip001.mp4")  # 直接放在一个文件夹中
            if not os.path.exists(out_temp):
                command = template2.format(vid_path, 0 ,duration, out_temp)
                subprocess.call(command, shell=True)
        else:
            duration_count = int(duration // args.duration_seed)
            duration_remain = duration % args.duration_seed  # 向下取整
            for i in range(duration_count):  # 看有几个duration_seed
                out_temp = os.path.join(out_path, f'{basename}_clip{i:03d}.mp4')

                if not os.path.exists(out_temp):
                    command = template2.format(vid_path, i*args.duration_seed, args.duration_seed, out_temp)
                    subprocess.call(command, shell=True)

            # if duration_remain != 0.:
            #     out_temp = os.path.join(out_path, f'{basename}_{duration_count:03d}.mp4')
            #     if not os.path.exists(out_temp):
            #         command = template2.format(vid_path, duration_count*args.duration_seed, duration, out_temp)
            #         subprocess.call(command, shell=True)


# 处理单个视频
def clip_vid_single_youtube(data, vidfile):
   # 获取持续时间列表，可能需要考虑如何避免在每个进程中重复读取
    root_dir, out_dir = args.root_dir, args.out_dir  # 从全局变量或传递的参数获取这些值
    basename = os.path.splitext(vidfile)[0]
    vid_path = os.path.join(root_dir, vidfile)
    duration = data[vid_path]
    out_path = os.path.join(out_dir, basename)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if duration <= args.duration_seed:  # 第一种情况是总时长小于持续的时间时，这个地方应该不存在，但是为了检验保留一步
        print("less than duration_seed")
        count += 1
        out_temp= os.path.join(out_path, f"{basename}_clip001.mp4")  # 直接放在一个文件夹中
        if not os.path.exists(out_temp):
            command = template2.format(vid_path, 0 ,duration, out_temp)
            subprocess.call(command, shell=True)
    else:
        duration_count = int(duration // args.duration_seed)
        duration_remain = duration % args.duration_seed  # 向下取整
        for i in range(duration_count):  # 看有几个duration_seed
            out_temp = os.path.join(out_path, f'{basename}_clip{i:03d}.mp4')

            if not os.path.exists(out_temp):
                command = template2.format(vid_path, i*args.duration_seed, args.duration_seed, out_temp)
                subprocess.call(command, shell=True)

            if duration_remain >= 30.:
                out_temp = os.path.join(out_path, f'{basename}_clip{duration_count:03d}.mp4')
                if not os.path.exists(out_temp):
                    command = template2.format(vid_path, duration_count*args.duration_seed, duration, out_temp)
                    subprocess.call(command, shell=True)

def clip_multi():
    files = os.listdir(args.root_dir)
    data = get_duration(args.json_file)
    func = partial(clip_vid_single_youtube, data) 
    with Pool(processes=32) as pool:
        list(tqdm(pool.imap(func, files), total=len(files)))


# 使用concurrent的写法
def process_video(vid_file, json_data):
    # 假设 `json_data` 是提前加载的视频持续时间数据
    basename = os.path.splitext(vid_file)[0]
    vid_path = os.path.join(args.root_dir, vid_file)
    duration = json_data[vid_path]  # 假设这样可以获取到持续时间
    out_path = os.path.join(args.out_dir, basename)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if duration <= args.duration_seed:  # 第一种情况是总时长小于持续的时间时，这个地方应该不存在，但是为了检验保留一步
        print("less than duration_seed")
        count += 1
        out_temp= os.path.join(out_path, f"{basename}_001.mp4")  # 直接放在一个文件夹中
        if not os.path.exists(out_temp):
            command = template2.format(vid_path, 0 ,duration, out_temp)
            subprocess.call(command, shell=True)
    else:
        duration_count = int(duration // args.duration_seed)
        duration_remain = duration % args.duration_seed  # 向下取整
        for i in range(duration_count):  # 看有几个duration_seed
            out_temp = os.path.join(out_path, f'{basename}_{i:03d}.mp4')

            if not os.path.exists(out_temp):
                command = template2.format(vid_path, i*args.duration_seed, args.duration_seed, out_temp)
                subprocess.call(command, shell=True)

            if duration_remain != 0.:
                out_temp = os.path.join(out_path, f'{basename}_{duration_count:03d}.mp4')
                if not os.path.exists(out_temp):
                    command = template2.format(vid_path, duration_count*args.duration_seed, duration, out_temp)
                    subprocess.call(command, shell=True)
    return basename  # 假设返回视频的基本名以更新进度条

# def update_progress(future):
#     result = future.result()  # 获取 process_video 函数的返回结果
#     pbar.update(1)  # 每完成一个视频文件，进度条更新一次


if __name__=="__main__":

    clip_multi() 

    # concurrent用法
    # data = get_duration(args.json_file)  # 提前加载持续时间数据
    # files = os.listdir(args.root_dir)
    # with tqdm(total=len(files)) as pbar:  # 初始化进度条
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         futures = [executor.submit(process_video, file, data) for file in files]
    #         for future in concurrent.futures.as_completed(futures):
    #             future.add_done_callback(update_progress)  # 为每个 future 添加回调函数以更新进度条
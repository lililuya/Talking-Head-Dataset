import moviepy.editor as mp
from pydub import AudioSegment
import os
import json


frame_path = "/mnt/sdb/cxh/liwen/DATA/lrs2_preprocessed"
# video_path = "/mnt/sdb/liwen/wav2lip_288x288/video_clip2"

# file_path = "/mnt/sdb/liwen/wav2lip_288x288/fps_avspeech.json"

file_lsr2_path = "/mnt/sdb/liwen/wav2lip_288x288/fps_lsr2.txt"
ill_file_path = "/mnt/sdb/liwen/wav2lip_288x288/ill_avspeech.txt"
# fps_file = ""


def get_audio_length(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio_length = len(audio) / 1000  # 将毫秒转换为秒
    return audio_length


def calculate_video_fps(vid_file):
    video = mp.VideoFileClip(vid_file)
    video_fps = video.fps
    video_duration = video.duration
    video.close()
    return video_fps, video_duration

def cal_synchronize(audio_path, frame_path):
    audio_len = get_audio_length(audio_path)
    frame = os.listdir(frame_path)
    frame_len = len(frame)
    real_fps = frame_len/audio_len
    return real_fps


def cal_ill_fps_txt(file_path):
    count = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # print(line)
            line = line.strip()
            if float(line)<=20 or float(line)>=30:
                count += 1
    return count

def cal_ill_fps_json(file_path):
    count = 0 
    with open(file_path,'r') as file:
        data = json.load(file)
        print(type(data))
        for item in data.values():
            if float(item)<=20 or float(item)>=30:
                count += 1 
    return count


def get_fps_json(root_dir = "/mnt/sda/cxh/data/avspeech_frames_re"):
    fps_file = {}
    for par_item in os.listdir(root_dir):
        base_path = os.path.join(root_dir, par_item)
        for base_item in os.listdir(base_path):
            clip_path = os.path.join(base_path, base_item)
            # print(clip_path)
            audio_exist = os.path.join(clip_path, "audio.wav")
            if os.path.isfile(audio_exist):
                audio_path = audio_exist
                frame_path = clip_path
                fps = cal_synchronize(audio_path, frame_path)
                fps_file[clip_path] = fps
            else:
                print("missing wav")       
    return fps_file

def write_json(json_path, fps_file):
    with open(json_path, 'w') as json_file:
        json.dump(fps_file, json_file)

def read_file():
    pass

if __name__ =="__main__":
    file_json = get_fps_json()
    json_path = "/mnt/sdb/cxh/liwen/wav2lip_288x288/tools/file/avspeech_real_fps.json"
    write_json(json_path, file_json)
    # count = cal_ill_fps_json(file_path)
    # print(count)



                
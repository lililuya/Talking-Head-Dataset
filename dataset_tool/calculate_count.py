import os
import subprocess
from tqdm import tqdm
import sys

root = "/mnt/hd/data/MEAD"
template = "tar -xf {} -C {}"


def calculate_count():
    for item in os.listdir(root):
        path = os.path.join(root, item)
        
def tar_file(root):
    for item in tqdm(sorted(os.listdir(root)), "process:"):
        item_path = os.path.join(root, item)
        for tar in os.listdir(item_path):
            if tar.endswith(".tar"):
                tar_path = os.path.join(item_path, tar)
                out_path = os.path.dirname(tar_path)
                command = template.format(tar_path, out_path)
                subprocess.call(command, shell=True)
            
if __name__=="__main__":
    tar_file(root)
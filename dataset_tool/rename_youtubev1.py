import os

# Youtubev1-000.mp4
def rename_mp4_file(directory):
    for filename in os.listdir(directory):
        if filename.startswith('Youtubev1-') and filename.endswith('.mp4'):
            new_filename = filename.replace('Youtubev1-', '')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        

if __name__=="__main__":
    directory = "/mnt/hd/data/YouTube_25"
    rename_mp4_file(directory)
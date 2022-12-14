import os
import threading

NUM_THREADS = 20
VIDEO_ROOT = r'data\ucf5\videos\test'   # Directory for videos
FRAME_ROOT = r'data\ucf5\frames\test'  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%05d.jpg'):
    os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf scale=256:256 ' f'{FRAME_ROOT}/{video[:-4]}/{tmpl}')


def target(video_list):
    # video[:-4] means substract .avi
    for video in video_list:
        video_path = os.path.join(FRAME_ROOT, video[:-4])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        extract(video)


if not os.path.exists(VIDEO_ROOT):
    raise ValueError('Please download videos and set VIDEO_ROOT variable.')
if not os.path.exists(FRAME_ROOT):
    os.makedirs(FRAME_ROOT)

video_list = os.listdir(VIDEO_ROOT)
splits = list(split(video_list, NUM_THREADS))

threads = []
for i, split in enumerate(splits):
    thread = threading.Thread(target=target, args=(split,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
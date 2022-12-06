'''
If source video is not at the target fps, convert it and owerwrite to the original video.
'''
from moviepy.editor import VideoFileClip
import configargparse
import os
from os import path

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--source_name", type=str)
    parser.add_argument("--video_folder", type=str)
    parser.add_argument("--target_fps", type=float, default=25.0)
    return parser

if __name__ == '__main__':
    # load the configs
    parser = config_parser()
    args = parser.parse_args()

    # check if the video is mp4 or avi
    if path.isfile(f"{args.video_folder}/{args.source_name}.mp4"): video_path = f"{args.video_folder}/{args.source_name}.mp4"
    elif path.isfile(f"{args.video_folder}/{args.source_name}.avi"): video_path = f"{args.video_folder}/{args.source_name}.avi"
    else: raise NotImplementedError("Source can only be a video with .mp4 or .avi extension")

    # read the video
    clip = VideoFileClip(video_path)

    # if not target fps, convert
    if clip.fps != args.target_fps:
        clip = clip.set_fps(args.target_fps)
        
        # write converted video with temporary name
        temp_name = f"{args.video_folder}/{args.source_name}_temp.mp4"
        clip.write_videofile(temp_name, audio_codec='aac', codec='h264')
        
        # remove the video with non-target fps
        os.remove(video_path)

        # rename the converted video to actual name
        os.rename(temp_name, f"{args.video_folder}/{args.source_name}.mp4")

    else: print("The video is already at the target fps, skipping the conversion")
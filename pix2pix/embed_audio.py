'''
Embed target audio to the soundless synthetic video. Remove soundless video.
'''
import moviepy.editor as mpe
import configargparse
import os

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--video_folder", type=str)
    parser.add_argument("--source_name", type=str)
    parser.add_argument("--target_name", type=str)
    
    return parser

if __name__ == '__main__':

    # load the configs
    parser = config_parser()
    args = parser.parse_args()    

    soundless_video_path = f"{args.video_folder}/source_{args.source_name}_target_{args.target_name}.mp4"
    soundless_video = mpe.VideoFileClip(soundless_video_path)
    target_audio = mpe.AudioFileClip(f"{args.video_folder}/{args.target_name}.wav")
   
    final_video = soundless_video.set_audio(target_audio)
    final_video_path = f"{args.video_folder}/source_{args.source_name}_target_{args.target_name}_with_audio.mp4"
    final_video.write_videofile(final_video_path, audio_codec='aac', codec='h264')

    os.remove(str(soundless_video_path))


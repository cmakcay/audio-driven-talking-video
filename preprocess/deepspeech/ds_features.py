'''
Extracts deepspeech features from audio using deepspeech frozen model. Input audio is processed to
be in the required format, as explained in following link: 
 https://docs.openvino.ai/latest/omz_models_model_mozilla_deepspeech_0_8_2.html
'''

import configargparse
from os import path
from moviepy.editor import VideoFileClip, AudioFileClip
from scipy.io import wavfile
import resampy
from python_speech_features import mfcc
import tensorflow as tf
from tensorflow.python.framework.ops import get_default_graph, reset_default_graph
import numpy as np

# # # Define Constants # # #
# "along with the current frame, the network expects 9 preceding frames and 9 succeeding frames. The absent context frames are filled with zeros."
# "26 MFCC coefficients per each frame"
# DeepSpeech requires sample frequency of 16kHz
# Default deepspeech frame rate is 50
CONTEXT_FRAMES = 9
MFCC_COEFFS = 26
AUDIO_SAMPLE_RATE = 16000
DEEPSPEECH_FRAME_RATE = 50.
FEATURES_WINDOW_SIZE = 16

def get_ds(args):
    
    # # # Read media file # # #
    # audio source can be from source or target and it can ve a video or audio file
    media_name = args.target_name if args.inference else args.source_name
    media_path = f"{args.video_folder}/{media_name}"

    # audio input
    if path.isfile(f"{media_path}.wav"): pass # already wav, do nothing
    elif path.isfile(f"{media_path}.mp3"): AudioFileClip(f"{media_path}.mp3").write_audiofile(f"{media_path}.wav",codec='pcm_s16le') # convert mp3 to wav 

    # video input 
    else:
        if path.isfile(f"{media_path}.avi"): media_with_extension = f"{media_path}.avi"
        elif path.isfile(f"{media_path}.mp4"): media_with_extension = f"{media_path}.mp4"
        else: raise NotImplementedError("Media can only be an audio with .wav or .mp3 extension or a video with .mp4 or .avi extension")

        # extract audio from video
        VideoFileClip(media_with_extension).audio.write_audiofile(f"{media_path}.wav",codec='pcm_s16le')
    
    # # # Now we have the desired audio with .wav extension # # #
    rate, data = wavfile.read(f"{media_path}.wav") # sample rate and data(audio)
    audio = np.copy(data)

    # data is n_samples, n_channels. only get the first channel if multiple channels are present
    data = data[:,0]
    data = resampy.resample(data.astype(float), rate, AUDIO_SAMPLE_RATE)

    # get mfcc features
    mfcc_features = mfcc(signal=data.astype("int16"), samplerate=AUDIO_SAMPLE_RATE, numcep=MFCC_COEFFS)

    # Skip one of two features as rnn stride of deepspeech is 2
    mfcc_features = mfcc_features[::2]

    # zero padding at start and end for stride trick
    pad = np.zeros((CONTEXT_FRAMES, MFCC_COEFFS), dtype=mfcc_features.dtype)
    mfcc_features_padded = np.concatenate((pad, mfcc_features, pad))
    
    # strided array to make shape x, 19, 26
    window_size = 2 * CONTEXT_FRAMES + 1 # [9 previous 1 current 9 next]
    strided_features = np.lib.stride_tricks.as_strided(
        mfcc_features_padded,
        shape=(mfcc_features.shape[0], window_size, MFCC_COEFFS),
        strides=(mfcc_features_padded.strides[0], mfcc_features_padded.strides[0], mfcc_features_padded.strides[1]),
        writeable=False)

    # make the array 2 dimensional
    strided_features = np.reshape(strided_features, (mfcc_features.shape[0], -1))

    # normalize the features
    strided_features = np.copy(strided_features)
    graph_input = (strided_features - np.mean(strided_features)) / np.std(strided_features)

    # # # Now we have the input to run inference # # #

    with tf.io.gfile.GFile(args.deepspeech_model_path, "rb") as f:
        graph = get_default_graph()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
        tf.import_graph_def(graph_def, name="deepspeech")
        
        input_node = graph.get_tensor_by_name('input_node:0')
        input_length = graph.get_tensor_by_name('input_lengths:0')
        logits = graph.get_tensor_by_name('logits:0')       

        with tf.compat.v1.Session(graph=graph) as sess:
            
            # run inference
            network_output = sess.run(logits, feed_dict={input_node: graph_input[np.newaxis, ...],
                                                            input_length: [graph_input.shape[0]]})
        
            # # # Interpolate features for target fps value # # #
            # get audio duration in seconds
            audio_duration = float(audio.shape[0]) / rate

            # for source video, we want number of deepspeech features to be exactly same as number of frames
            # as we get person specific features. so, we get number of frames beforehand. with inference
            # we do not have number of frames (because we can have audio) and we do not need to be exact
            if args.num_frames == None:
                num_frames = int(round(audio_duration * args.target_fps))
            else:
                num_frames = args.num_frames

            output = network_output[:, 0] # output is (x, 29) 
            num_features = output.shape[1]

            input_tags = np.arange(output.shape[0]) / DEEPSPEECH_FRAME_RATE
            interpolated_tags = np.arange(num_frames) / args.target_fps
            interpolated_features = np.zeros((num_frames, num_features))
            for feature in range(num_features):
                interpolated_features[:, feature] = np.interp(interpolated_tags, input_tags, output[:, feature])

            # # # Now we have the interpolated features, window the features to have 16x29 for audio2exp network # # # 
            
            # zero padding the features  
            pad = np.zeros((int(FEATURES_WINDOW_SIZE / 2), num_features))
            padded_interpolated_features = np.concatenate((pad, interpolated_features, pad))
            windowed_output = []
            for i in range(padded_interpolated_features.shape[0]-FEATURES_WINDOW_SIZE):
                windowed_output.append(padded_interpolated_features[i:i+FEATURES_WINDOW_SIZE])

            ds_features = np.array(windowed_output)

            # # # Finally, we have the deepspeech features, save them # # #  
            for i, ds in enumerate(ds_features):
                np.save(f"{args.ds_output_path}/{i}.npy", ds)

        reset_default_graph()
# Audio Driven Speech Mimicry 
This repository contains the code to manipulate a source input video of a speaking person to mimic the audio from a given target video.

This code is tested with Ubuntu operating system and Python 3.7.

## Requirements (only first time using the code)
1. First, a virtual environment should be created, e.g. using venv by:

```
python -m venv venv
```
The virtual environment is activated by:

```
source venv/bin/activate
```

2.  After activating the virtual environment, run the following to install the requirements.

```
pip install --upgrade pip
pip install -r requirements.txt
```

3. Create a directory named "Documents" inside the folder, which will contain some necessary files to run the pipeline. Create a subdirectory in this directory named "pipeline_files". Download [FLAME 2020](https://flame.is.tue.mpg.de) and move "generic_model.pkl" to Documents/pipeline_files/flame_files. Download "landmark_embedding.npy" from [DECA](https://github.com/YadiraF/DECA/tree/master/data) and move it to the "flame_files" as well. Download deepspeech trained [model](https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz), and move "output_graph.pb" to Documents/pipeline_files/trained_models. Finally, download trained tracker [model](https://drive.google.com/file/d/1xyIDFreHW_f_uBmAydBxmJFzeO4WdVfh/view?usp=sharing) and audio2exp [model](https://drive.google.com/file/d/1Bs888B21GXVTUXMqzQQmOpUnZLTO397E/view?usp=sharing) and move "tracker.pt" and "a2e.pt" to "trained_models" as well.

## Run
1. Training the source <br />
To use a video as source (i.e. the actor), the model should be trained for that video first. Training the model for source video is initated by running the following inside the directory.
```
source venv/bin/activate
./train_source.sh SOURCE_NAME
```
Where SOURCE_NAME is the name (without extension) of the source video. The source video should be located in the "Documents/video" folder. 

2. Creating the fake video (inference) <br />
Generation of the fake video is initated by running the following inside the directory.
```
source venv/bin/activate
./create_fake_video.sh SOURCE_NAME TARGET_NAME
```

Where SOURCE_NAME and TARGET_NAME are the names (without extension) of the source and target videos respectively. The videos shouls be located in the "Documents/video" folder. For example, if the source video is "aa.mp4" and target video is "bb.mp4", you should run:
```
./create_fake_video.sh aa bb
```
The target can also be an audio file. This will get the speech from the target video/audio and manipulate the source video based on this speech. 

When the run is complete, the resulting fake video will be inside the "Documents/video" folder.

## Tuning
In preprocess/ds_to_flame_params.py, there are two parameters "jaw_gain" and "jaw_closure". jaw_gain determines how much the mouth opens given the speech. Higher gain leads to more movement at the mouth. jaw_closure determines how much the mouth should be closed during silence. By default, there is an offset in mouth in silence, so it aims to solve this. Current parameters seem to work well, but it can be experimented with other parameters to test the effects.
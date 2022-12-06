#! /bin/sh

BASE=`pwd`
SOURCE_NAME=$1
echo "SOURCE_NAME: $1"

# _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# Define the directories

PIPELINE_FILES_PATH=$BASE/Documents/pipeline_files
VIDEO_FOLDER=$BASE/Documents/video
DATASET_PATH=$BASE/Documents/dataset
# _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_


# Do not change the following code

cd ./preprocess

# Convert the source video to target fps, if it is not already 25 fps
python convert_fps.py --source_name $SOURCE_NAME --video_folder $VIDEO_FOLDER

# Extract FLAME parameters
python extract_flame_params.py --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME --video_folder $VIDEO_FOLDER

# Get deepspeech features
python get_deepspeech_features.py --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME --video_folder $VIDEO_FOLDER

# Get person specific mapping
python get_person_specific.py --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME

# Now create edge maps and masks
python create_edge_maps.py --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME

cd ../pix2pix
# We have training dataset now, start training
python train.py --name $SOURCE_NAME --video_folder $VIDEO_FOLDER --dataroot $DATASET_PATH --checkpoints_dir "$PIPELINE_FILES_PATH/pix2pix/checkpoints/" --model edgemap --dataset_mode edgemap --no_flip --batch_size 1 --no_html 

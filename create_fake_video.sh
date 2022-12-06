#! /bin/sh

BASE=`pwd`
SOURCE_NAME=$1
TARGET_NAME=$2
echo "SOURCE_NAME: $1"
echo "TARGET_NAME: $2"

# _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_
# Define the directories

PIPELINE_FILES_PATH=$BASE/Documents/pipeline_files
VIDEO_FOLDER=$BASE/Documents/video
DATASET_PATH=$BASE/Documents/dataset
# _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_


# Do not change the following code

cd ./preprocess

# First get deepspeech features of target video
python get_deepspeech_features.py --inference --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME --target_name $TARGET_NAME --video_folder $VIDEO_FOLDER

# Get flame params from deepspeech features
python ds_to_flame_params.py --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME --target_name $TARGET_NAME

# Now create edge maps and masks
python create_edge_maps.py --inference --dataset_path $DATASET_PATH --pipeline_files_path $PIPELINE_FILES_PATH --source_name $SOURCE_NAME --target_name $TARGET_NAME

cd ../pix2pix
# We have inference dataset, we can run inference now
python inference.py --name $SOURCE_NAME  --video_folder $VIDEO_FOLDER --dataroot $DATASET_PATH --target_name $TARGET_NAME --model edgemapinference --dataset_mode edgemapinference --epoch "latest" --checkpoints_dir "$PIPELINE_FILES_PATH/pix2pix/checkpoints/"

# Finally add target audio to video
python embed_audio.py --video_folder $VIDEO_FOLDER --source_name $SOURCE_NAME --target_name $TARGET_NAME
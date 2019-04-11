#!/bin/bash -e

pip_packages="nose numpy torch torchvision scikit-image tensorboardX"

apt-get update
apt-get install -y wget ffmpeg git

pushd ../..

source qa/setup_dali_extra.sh

cd docs/examples/video

mkdir -p video_files


container_path=${DALI_EXTRA_PATH}/db/optical_flow/sintel_trailer/sintel_trailer.mp4

IFS='/' read -a container_name <<< "$container_path"
IFS='.' read -a splitted <<< "${container_name[-1]}"

for i in {0..4};
do
    ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_path -vcodec copy -acodec copy -y video_files/${splitted[0]}_$i.${splitted[1]} 
done

cd superres_pytorch

DATA_DIR=data_dir/720p/scenes
# Creating simple working env for PyTorch SuperRes example
mkdir -p $DATA_DIR/train/
mkdir -p $DATA_DIR/val/


cp ../video_files/* $DATA_DIR/train/
cp ../video_files/* $DATA_DIR/val/

# Pre-trained FlowNet2.0 weights
FLOWNET_PATH=/data/dali/pretrained_models/FlowNet2-SD_checkpoint.pth.tar

git clone https://github.com/NVIDIA/flownet2-pytorch.git

cd ..

test_body() {
    # test code
    # First running simple code
    python video_example.py

    nosetests --verbose ../../../dali/test/python/test_video_pipeline.py

    cd superres_pytorch

    python main.py --loader DALI --rank 0 --batchsize 2 --frames 3 --root $DATA_DIR --world_size 1 --is_cropped --max_iter 100 --min_lr 0.0001 --max_lr 0.001 --crop_size 512 960 --flownet_path $FLOWNET_PATH

    cd ..
}

source ../../../qa/test_template.sh

popd

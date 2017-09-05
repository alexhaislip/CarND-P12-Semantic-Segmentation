#!/bin/bash
set -ex
pushd data
wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
unzip data_road.zip
popd
pip install -U tqdm


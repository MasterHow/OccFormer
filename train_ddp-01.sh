#!/bin/bash
source /opt/conda/bin/activate
conda activate occformer
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
cd /workspace/mnt/storage/shihao/MyCode-02/OccFormer/mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ..
pip install -r docs/requirements.txt
pip install yapf==0.40.1
apt-get install libgl1-mesa-glx -y
chmod 777 ./tools/dist_train.sh
bash ./tools/dist_train.sh ./projects/configs/occformer_kitti/occformer_kitti_PL-VoxOn.py 8

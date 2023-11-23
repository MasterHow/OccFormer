#!/bin/bash
source /opt/conda/bin/activate
conda create -n occformer python=3.7 -y
conda activate occformer
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install --no-index --find-links=/workspace/mnt/storage/shihao/MyCode-02/OccFormer/pip_packages torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1
pip install --no-index --find-links=/workspace/mnt/storage/shihao/MyCode-02/OccFormer/pip_packages openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
cd /workspace/mnt/storage/shihao/MyCode-02/OccFormer/mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ..
pip install -r docs/requirements.txt
pip install yapf==0.40.1
chmod 777 ./tools/dist_train.sh
bash ./tools/dist_train.sh ./projects/configs/occformer_kitti/occformer_kitti.py 8

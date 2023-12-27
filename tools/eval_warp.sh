#!/usr/bin/env bash

# 检查是否提供了预测根目录参数
if [ -z "$1" ]; then
    echo "请提供预测根目录作为参数。"
    exit 1
fi

predict_root="$1"

# 定义其他重复使用的参数
list_dir="${predict_root}/sequences/08/bev_vis/"
flow_dir="${predict_root}/sequences/08/bev_flow/"

source /opt/conda/bin/activate
conda activate mayavi
cd /workspace/mnt/storage/shihao/MyCode-02/OccFormer

# 使用变量来简化命令 bev可视化
xvfb-run -a python tools/bev_view_gen.py --dset_root=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti --pred_root="$predict_root"

conda deactivate
conda activate mmflow
cd /workspace/mnt/storage/shihao/MyCode-02/OccFormer/fast_blind_video_consistency

# 使用变量来简化命令
python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"

#### 临时，用于生成伪光流真值
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/00/bev_vis/"
#flow_dir="${predict_root}/sequences/00/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/01/bev_vis/"
#flow_dir="${predict_root}/sequences/01/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/02/bev_vis/"
#flow_dir="${predict_root}/sequences/02/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/03/bev_vis/"
#flow_dir="${predict_root}/sequences/03/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/04/bev_vis/"
#flow_dir="${predict_root}/sequences/04/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/05/bev_vis/"
#flow_dir="${predict_root}/sequences/05/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/06/bev_vis/"
#flow_dir="${predict_root}/sequences/06/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/07/bev_vis/"
#flow_dir="${predict_root}/sequences/07/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/09/bev_vis/"
#flow_dir="${predict_root}/sequences/09/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
## 定义其他重复使用的参数
#list_dir="${predict_root}/sequences/10/bev_vis/"
#flow_dir="${predict_root}/sequences/10/bev_flow/"
## 使用变量来简化命令
#python compute_flow_occlusion_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#python evaluate_WarpError_mmflow.py -list_dir "$list_dir" -flow_dir "$flow_dir"
#### 临时，用于生成伪光流真值

# 还原工作目录
cd "$flow_dir"
rm -rf fw_flow bw_flow fw_flow_rgb bw_flow_rgb fw_occlusion bw_occlusion
cd /workspace/mnt/storage/shihao/MyCode-02/OccFormer/

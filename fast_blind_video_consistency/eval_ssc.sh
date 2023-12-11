#!/bin/bash
# GT seq00
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq01
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq02
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq03
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq04
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq05
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq06
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq07
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq08
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq09
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# GT seq10
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_flow_gt/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_vis/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_flow_gt/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_flow_gt/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

#######################################################################################################################
# PS seq00
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/00/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq01
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/01/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq02
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/02/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq03
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/03/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq04
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/04/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq05
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/05/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq06
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/06/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq07
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/07/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq08
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/08/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq09
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/09/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency

# PS seq10
python compute_flow_occlusion_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_flow_pseudo/
python evaluate_WarpError_mmflow.py -list_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_vis_pseudo/ -flow_dir /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_flow_pseudo/
cd /workspace/mnt/storage/shihao/BEV_Flow/SemanticKITTI/data_odometry_voxels/dataset/sequences/10/bev_flow_pseudo/
rm -rf fw_flow/
cd /workspace/mnt/storage/shihao/MyCode-01/fast_blind_video_consistency


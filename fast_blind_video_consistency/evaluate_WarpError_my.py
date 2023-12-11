#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')
    

    ### testing options
    parser.add_argument('-task',            type=str,     required=False,            help='evaluated task')
    parser.add_argument('-method',          type=str,     required=False,            help='test model name')
    parser.add_argument('-dataset',         type=str,     required=False,            help='test datasets')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-flow_dir', type=str, default='lists', help='path of the saved flow/occ/')
    parser.add_argument('-redo',            action="store_true",                    help='redo evaluation')

    opts = parser.parse_args()
    opts.cuda = True

    print(opts)

    # 我们直接把output放到对应的flow文件夹下
    # output_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset)
    output_dir = os.path.join(opts.flow_dir, "E_warp_result")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ## print average if result already exists
    metric_filename = os.path.join(output_dir, "WarpError.txt")
    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists, skip..." %metric_filename)

        cmd = 'tail -n1 %s' %metric_filename
        utils.run_cmd(cmd)
        sys.exit()

    ## flow warping layer
    device = torch.device("cuda" if opts.cuda else "cpu")
    flow_warping = Resample2d().to(device)

    # ### load video list
    # # 因为是整个文件夹都要测试，所以video_list直接用文件夹的名字读取就好了
    # # list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    # # with open(list_filename) as f:
    # #     video_list = [line.rstrip() for line in f.readlines()]
    # # list_dir就是根目录的路径
    # video_list = []
    # for root, dirs, files in os.walk(opts.list_dir, topdown=False):
    #     # 获取文件夹名字
    #     video_list = dirs

    # ### start evaluation
    # err_all = np.zeros(len(video_list))
    #
    # for v in range(len(video_list)):

    # video = video_list[v]

    # frame_dir直接用根目录+视频名字就可以了
    # frame_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset, video)
    # frame_dir = os.path.join(opts.list_dir, video)
    frame_dir = opts.list_dir

    # 直接在同一个目录下读取生成的光流、遮挡、可视化的光流
    # occ_dir = os.path.join(opts.data_dir, opts.phase, "fw_occlusion", opts.dataset, video)
    # flow_dir = os.path.join(opts.data_dir, opts.phase, "fw_flow", opts.dataset, video)
    # occ_dir = os.path.join(opts.flow_dir, "fw_occlusion", video)
    # flow_dir = os.path.join(opts.flow_dir, "fw_flow", video)
    occ_dir = os.path.join(opts.flow_dir, "fw_occlusion")
    flow_dir = os.path.join(opts.flow_dir, "fw_flow")

    # 我们的图像是png格式
    # frame_list = glob.glob(os.path.join(frame_dir, "*.jpg"))
    frame_list = glob.glob(os.path.join(frame_dir, "*.png"))

    err = 0
    for t in range(1, len(frame_list)):

        ### load input images
        # 我们是png格式的哦
        # filename = os.path.join(frame_dir, "%05d.jpg" %(t - 1))
        # filename = os.path.join(frame_dir, "%05d.png" % (t - 1))
        filename = os.path.join(frame_dir, "%06d.png" % (t - 1))
        img1 = utils.read_img(filename)
        # filename = os.path.join(frame_dir, "%05d.jpg" %(t))
        # filename = os.path.join(frame_dir, "%05d.png" % (t))
        filename = os.path.join(frame_dir, "%06d.png" % (t))
        img2 = utils.read_img(filename)

        # 更改提示语句
        # print("Evaluate Warping Error on %s-%s: video %d / %d, %s" %(opts.dataset, opts.phase, v + 1, len(video_list), filename))
        # print("Evaluate Warping Error on: video %d / %d, %s" % (v + 1, len(video_list), filename))
        print("Evaluate Warping Error on: video %s, %s" % (opts.list_dir, filename))


        ### load flow
        # filename = os.path.join(flow_dir, "%05d.flo" %(t-1))
        filename = os.path.join(flow_dir, "%06d.flo" % (t - 1))
        flow = utils.read_flo(filename)

        ### load occlusion mask
        # filename = os.path.join(occ_dir, "%05d.png" %(t-1))
        filename = os.path.join(occ_dir, "%06d.png" % (t - 1))
        occ_mask = utils.read_img(filename)
        noc_mask = 1 - occ_mask

        with torch.no_grad():

            ## convert to tensor
            img2 = utils.img2tensor(img2).to(device)
            flow = utils.img2tensor(flow).to(device)

            ## warp img2
            warp_img2 = flow_warping(img2, flow)

            ## convert to numpy array
            warp_img2 = utils.tensor2img(warp_img2)


        ## compute warping error
        diff = np.multiply(warp_img2 - img1, noc_mask)

        N = np.sum(noc_mask)
        if N == 0:
            N = diff.shape[0] * diff.shape[1] * diff.shape[2]

        err += np.sum(np.square(diff)) / N

    # err_all[v] = err / (len(frame_list) - 1)
    err_all = err / (len(frame_list) - 1)


    print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    err_all = np.append(err_all, err_all.mean())
    print("Save %s" %metric_filename)
    np.savetxt(metric_filename, err_all, fmt="%f")

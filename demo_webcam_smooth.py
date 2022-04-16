# coding: utf-8


import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render, render_filter
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    



    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_ver_sparse = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d', '2d_hybrid', 'filter')
    sparse_flag = args.opt in ('2d_sparse', '2d_hybrid')
    pre_ver = None
    i = 0
    while True:
        _ , frame_bgr = cap.read()
        # frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i % 20 == 0:
            n_pre, n_next = args.n_pre, args.n_next
            n = n_pre + n_next + 1
            queue_ver = deque()
            queue_ver_sparse = deque()
            queue_frame = deque()
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)
            if len(boxes) == 0:
                cv2.imshow('image', frame_bgr)
                k = cv2.waitKey(20)
                if (k & 0xff == ord('q')):
                    break
                continue
            
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            if sparse_flag and dense_flag:
                ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            else:
                ver_sparse = None
            # padding queue

            for _ in range(n_pre):
                queue_ver.append(ver.copy())
                if ver_sparse is not None:queue_ver_sparse.append(ver_sparse.copy())
            queue_ver.append(ver.copy())
            if ver_sparse is not None:queue_ver_sparse.append(ver_sparse.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())
        else:
                
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                if len(boxes) == 0:
                    cv2.imshow('image', frame_bgr)
                    k = cv2.waitKey(20)
                    if (k & 0xff == ord('q')):
                        break
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            if sparse_flag and dense_flag:
                ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            else:
                ver_sparse = None

            queue_ver.append(ver.copy())
            if ver_sparse is not None:queue_ver_sparse.append(ver_sparse.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            if ver_sparse is not None:ver_ave_sparse = np.mean(queue_ver_sparse, axis=0)
            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave[:,::args.dense_dist], size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            elif args.opt == 'filter':
                img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, filter=filter, alpha=0.7)
            elif args.opt == '2d_hybrid':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave[:,::args.dense_dist], size=1)
                img_draw = cv_draw_landmark(img_draw, ver_ave_sparse, size=2, color=(0, 0, 255), box = roi_box_lst[-1],is_dense = False)
            else:
                raise ValueError(f'Unknown opt {args.opt}')
            if args.opt2 == '3d':
                img_draw_3d = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
                img_draw_3d = cv2.flip(img_draw_3d,1)
                h,w, _ = img_draw_3d.shape
                cv2.imshow('image_3d',cv2.resize(img_draw_3d,(int(w*0.3), int(h*0.3))))
            img_draw = cv2.flip(img_draw,1)
            cv2.imshow('image', img_draw)
            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break

            queue_ver.popleft()
            if ver_sparse is not None:queue_ver_sparse.popleft()
            queue_frame.popleft()

        i += 1


if __name__ == '__main__':
    filter = cv2.imread('/home/tl/FaceMakeUp/3DDFA_V2/All_filter/9.png')
    mask = np.array(np.where(np.sum(filter,-1) == 255*3))
    filter_mask = np.ones_like(filter)
    for i in range(len(mask[0])):
        filter_mask[mask[0][i], mask[1][i]] = (0,0,0)
    filter = filter*filter_mask
    # filter = cv2.cvtColor(filter,cv2.COLOR_RGB2BGR)

    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode'),
    parser.add_argument('-o', '--opt', type=str, default='2d_hybrid', choices=['2d_sparse', '2d_dense', '3d','filter', '2d_hybrid'])
    parser.add_argument('-o2', '--opt2', type=str, default='none', choices=['3d', 'None'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-dense_dist', default=6, type=int, help='dense sampling distance')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)


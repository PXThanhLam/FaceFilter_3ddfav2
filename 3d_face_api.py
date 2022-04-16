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
from flask import Flask, jsonify, request 
import flask

import os




if __name__ == '__main__':
    app = Flask(__name__)
    list_filters = []
    for file in os.listdir('All_filter'):
        filter = cv2.imread('All_filter/' + file)
        mask = np.array(np.where(np.sum(filter,-1) == 255*3))
        filter_mask = np.ones_like(filter)
        for i in range(len(mask[0])):
            filter_mask[mask[0][i], mask[1][i]] = (0,0,0)
        filter = filter*filter_mask
        list_filters.append(filter)


    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode'),
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-dense_dist', default=6, type=int, help='dense sampling distance')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()

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
    
    
    tddfa.model.eval()
  
    n_pre, n_next = 1,1
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_ver_sparse = deque()
    queue_frame = deque()
    pre_ver = None
    frame_i = 0
    @app.route('/filter', methods=['GET'])
    def apply_filter():
        global n_pre
        global n_next
        global n
        global queue_ver
        global queue_ver_sparse
        global queue_frame
        global frame_i
        global pre_ver
        content = request.json
        
        img_src = cv2.imread(content['img_src'])
        filter_idx = content['filter_idx']
        if frame_i % 25 == 0:
            n_pre, n_next = 1,1
            n = n_pre + n_next + 1
            queue_ver = deque()
            queue_ver_sparse = deque()
            queue_frame = deque()
            boxes = face_boxes(img_src)
            if len(boxes) == 0:
                pre_ver = ver
                return jsonify({
               "result_path": "None",
             })

            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(img_src, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            param_lst, roi_box_lst = tddfa(img_src, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

            # refine
            param_lst, roi_box_lst = tddfa(img_src, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            # padding queue

            for _ in range(n_pre):
                queue_ver.append(ver.copy())
                queue_ver_sparse.append(ver_sparse.copy())
            queue_ver.append(ver.copy())
            queue_ver_sparse.append(ver_sparse.copy())

            for _ in range(n_pre):
                queue_frame.append(img_src.copy())
            queue_frame.append(img_src.copy())
        else:
                
            param_lst, roi_box_lst = tddfa(img_src, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(img_src)
                if len(boxes) == 0:
                    pre_ver = ver
                    return jsonify({
               "result_path": "None",
             })

                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(img_src, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            

            queue_ver.append(ver.copy())
            if ver_sparse is not None:queue_ver_sparse.append(ver_sparse.copy())
            queue_frame.append(img_src.copy())

        pre_ver = ver  # for tracking
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                    filter=list_filters[filter_idx], alpha=0.7)
            queue_ver.popleft()
            queue_ver_sparse.popleft()
            queue_frame.popleft()
            cv2.imwrite('res_filter.jpg', img_draw*255)
            return jsonify({'render_filter_path': '/home/tl/FaceMakeUp/3DDFA_V2/res_filter.jpg'})
        frame_i += 1
        return jsonify({'render_filter_path': 'not_ren_yet'})
    

    @app.route('/3d', methods=['GET'])
    def apply_3d():
        global n_pre
        global n_next
        global n
        global queue_ver
        global queue_ver_sparse
        global queue_frame
        global frame_i
        global pre_ver
        content = request.json
        
        img_src = cv2.imread(content['img_src'])
        if frame_i % 25 == 0:
            n_pre, n_next = 1,1
            n = n_pre + n_next + 1
            queue_ver = deque()
            queue_ver_sparse = deque()
            queue_frame = deque()
            boxes = face_boxes(img_src)
            if len(boxes) == 0:
                pre_ver = ver
                return jsonify({
               "result_path": "None",
             })

            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(img_src, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            param_lst, roi_box_lst = tddfa(img_src, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

            # refine
            param_lst, roi_box_lst = tddfa(img_src, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            # padding queue

            for _ in range(n_pre):
                queue_ver.append(ver.copy())
                queue_ver_sparse.append(ver_sparse.copy())
            queue_ver.append(ver.copy())
            queue_ver_sparse.append(ver_sparse.copy())

            for _ in range(n_pre):
                queue_frame.append(img_src.copy())
            queue_frame.append(img_src.copy())
        else:
                
            param_lst, roi_box_lst = tddfa(img_src, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(img_src)
                if len(boxes) == 0:
                    pre_ver = ver
                    return jsonify({
               "result_path": "None",
             })

                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(img_src, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
            ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            

            queue_ver.append(ver.copy())
            if ver_sparse is not None:queue_ver_sparse.append(ver_sparse.copy())
            queue_frame.append(img_src.copy())

        pre_ver = ver  # for tracking
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            queue_ver.popleft()
            queue_ver_sparse.popleft()
            queue_frame.popleft()
            cv2.imwrite('res_depth.jpg', img_draw)
            return jsonify({'render_3d': '/home/tl/FaceMakeUp/3DDFA_V2/res_depth.jpg'})
        frame_i += 1
        return jsonify({'render_3d': 'not_ren_yet'})

    app.run(debug=True, host='0.0.0.0', port = 3000)
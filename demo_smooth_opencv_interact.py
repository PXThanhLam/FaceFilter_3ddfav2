from tkinter import *
from PIL import ImageTk, Image
from numpy import pad
import threading

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

def show_msg1():
    global state
    stage = 1
    print(stage)
def show_msg2():
    global state
    stage = 2
    print(stage)
def show_msg3():
    global state
    stage = 3
    print(stage)
def show_msg4():
    global state
    stage = 4
    print(stage)
def show_msg5():
    global state
    stage = 5
    print(stage)
def show_msg6():
    global state
    stage = 6
    print(stage)
root = Tk()
root.title("Ella Demo")
root.geometry("1280x720")

top_frame = Frame(width=1024, height=580, background='red')       
top_frame.grid(row=0, column=0,padx = 100)

bottom_frame = Frame(width=1024, height=140, background='gold2')       
bottom_frame.grid(row=1, column=0) 



c1_label = Label(top_frame)

# c1_label = Label(top_frame, image =photo_big).grid( row=0, column=0)

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


state = 1
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 580)

def show_frame():
    # print('call func')
    global n_pre
    global n_next
    global n
    global queue_ver
    global queue_ver_sparse
    global queue_frame
    global frame_i
    global pre_ver
    _ , img_src = cap.read()
    # print(img_src.shape)
    is_call = False
    if frame_i % 100 == 0:
        n_pre, n_next = 1,1
        n = n_pre + n_next + 1
        queue_ver = deque()
        queue_ver_sparse = deque()
        queue_frame = deque()
        boxes = face_boxes(img_src)
        if len(boxes) == 0:
            img_src = cv2.flip(img_src,1)
            print('draw wrong 1')
            img_itk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)))
            c1_label.imgtk =  img_itk
            c1_label.configure(image=img_itk)
            c1_label.grid( row=0, column=0)
            c1_label.after(1, show_frame)
            is_call = True
            return 1

            # c1_label = Label(top_frame, image = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))).grid( row=0, column=0)

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
                img_src = cv2.flip(img_src,1)
                print('draw wrong 2')
                img_itk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)))
                c1_label.imgtk =  img_itk
                c1_label.configure(image=img_itk)
                c1_label.grid( row=0, column=0)
                c1_label.after(1, show_frame)
                is_call = True
                return 1

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
        ver_ave_sparse = np.mean(queue_ver_sparse, axis=0)
        if state == 0:
            img_draw = img_src
        elif state == 1:
            img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                filter=list_filters[0], alpha=0.7)
        elif state == 2:
            img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                filter=list_filters[1], alpha=0.7)
        elif state == 3:
            img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                filter=list_filters[2], alpha=0.7)
        elif state == 4:
            img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                filter=list_filters[3], alpha=0.7)
        elif state == 5:
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave[:,::args.dense_dist], size=1)
            img_draw = cv_draw_landmark(img_draw, ver_ave_sparse, size=2, color=(0, 0, 255), box = roi_box_lst[-1],is_dense = False)

        elif state == 6:
            img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
        # print('draw right')
        img_draw = np.array(img_draw*255,np.uint8)
        img_draw = cv2.flip(img_draw,1)
        img_itk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)))
        c1_label.imgtk =  img_itk
        c1_label.configure(image=img_itk)
        c1_label.grid( row=0, column=0)
        c1_label.after(1, show_frame)
        is_call = True
        queue_ver.popleft()
        queue_ver_sparse.popleft()
        queue_frame.popleft()
    frame_i += 1
    if not is_call:
        # print('draw first')
        img_src = cv2.flip(img_src,1)
        img_itk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)))
        c1_label.imgtk =  img_itk
        c1_label.configure(image=img_itk)
        c1_label.grid( row=0, column=0)
        c1_label.after(1, show_frame)

    
# show_frame()
image = Image.open('/home/tl/photo.jpg')
photo = ImageTk.PhotoImage(image.resize((180, 80), Image.ANTIALIAS))
button1 = Button(bottom_frame, image = photo, command=show_msg1).grid( row=1, column=0,padx=10)
button2 = Button(bottom_frame, image = photo, command=show_msg2).grid( row=1, column=1,padx=10)
button3 = Button(bottom_frame, image = photo, command=show_msg3).grid( row=1, column=2,padx=10)
button4 = Button(bottom_frame, image = photo, command=show_msg4).grid( row=1, column=3,padx=10)
button5 = Button(bottom_frame, image = photo, command=show_msg5).grid( row=1, column=4,padx=10)
button6 = Button(bottom_frame, image = photo, command=show_msg6).grid( row=1, column=5,padx=10)
print(button1)
root.mainloop()


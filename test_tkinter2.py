import sys
import cv2
import threading
import tkinter as tk
import tkinter.ttk as ttk
from queue import Queue
from PIL import Image
from PIL import ImageTk


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


import os
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
    tddfa.model.eval()
    face_boxes = FaceBoxes()


n_pre, n_next = 1,1
n = n_pre + n_next + 1
queue_ver = deque()
queue_ver_sparse = deque()
queue_frame = deque()
pre_ver = None
frame_i = 0

class App(tk.Frame):
    def __init__(self, parent, title):
        tk.Frame.__init__(self, parent)
        self.is_running = False
        self.state = 1
        self.thread = None
        self.queue = Queue()
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (1000, 600), "white"))
        parent.wm_withdraw()
        parent.wm_title(title)
        self.create_ui()
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        parent.wm_protocol("WM_DELETE_WINDOW", self.on_destroy)
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)
        parent.wm_deiconify()

    def create_ui(self):
        self.button_frame = ttk.Frame(self)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.RIGHT)
        self.start_button = ttk.Button(self.button_frame, text="Filter1", command=self.start)
        self.start_button.pack(side=tk.RIGHT)

        self.start1_button = ttk.Button(self.button_frame, text="Filter2", command=self.start1)
        self.start1_button.pack(side=tk.RIGHT)

        self.start2_button = ttk.Button(self.button_frame, text="Filter3", command=self.start2)
        self.start2_button.pack(side=tk.RIGHT)

        self.start3_button = ttk.Button(self.button_frame, text="Filter4", command=self.start3)
        self.start3_button.pack(side=tk.RIGHT)

        self.start4_button = ttk.Button(self.button_frame, text="Point", command=self.start4)
        self.start4_button.pack(side=tk.RIGHT)

        self.start5_button = ttk.Button(self.button_frame, text="Depth", command=self.start5)
        self.start5_button.pack(side=tk.RIGHT)


        self.view = ttk.Label(self, image=self.photo)
        self.view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def on_destroy(self):
        self.stop()
        self.after(20)
        if self.thread is not None:
            self.thread.join(0.2)
        self.winfo_toplevel().destroy()

    def start(self):
        self.is_running = True
        self.state = 1
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
    def start1(self):
        self.is_running = True
        self.state = 2
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
    def start2(self):
        self.is_running = True
        self.state = 3
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
    def start3(self):
        self.is_running = True
        self.state = 4
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
    def start4(self):
        self.is_running = True
        self.state = 5
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
    def start5(self):
        self.is_running = True
        self.state = 6
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()
    
    

    def stop(self):
        self.is_running = False

    def videoLoop(self, mirror=False):
        No=0#'/home/tl/FaceMakeUp/3DDFA_V2/fann.mp4'
        cap = cv2.VideoCapture(No)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        global n_pre
        global n_next
        global n
        global queue_ver
        global queue_ver_sparse
        global queue_frame
        global frame_i
        global pre_ver
        while self.is_running:
            # print(self.state)
            ret, to_draw = cap.read()
            if mirror is True:
                to_draw = to_draw[:,::-1]

            #######
            is_call = False
            img_src = to_draw
            if frame_i % 100 == 0:
                n_pre, n_next = 1,1
                n = n_pre + n_next + 1
                queue_ver = deque()
                queue_ver_sparse = deque()
                queue_frame = deque()
                boxes = face_boxes(img_src)
                if len(boxes) == 0:
                    is_call = True
                    img_src = cv2.flip(img_src,1)
                    image = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
                    self.queue.put(image)
                    self.event_generate('<<MessageGenerated>>')
                    print('draw wrong 1')
                    continue


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
                        img_src = cv2.flip(img_src,1)
                        image = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
                        self.queue.put(image)
                        self.event_generate('<<MessageGenerated>>')
                        is_call = True
                        continue

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
                if self.state == 0:
                    img_draw = img_src
                elif self.state == 1:
                    img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                        filter=list_filters[0], alpha=0.7)
                elif self.state == 2:
                    img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                        filter=list_filters[1], alpha=0.7)
                elif self.state == 3:
                    img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                        filter=list_filters[2], alpha=0.7)
                elif self.state == 4:
                    img_draw = render_filter(queue_frame[n_pre], [ver_ave], tddfa.tri, 
                                        filter=list_filters[3], alpha=0.7)
                elif self.state == 5:
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave[:,::args.dense_dist], size=1)
                    img_draw = cv_draw_landmark(img_draw, ver_ave_sparse, size=2, color=(0, 0, 255), box = roi_box_lst[-1],is_dense = False)/255

                elif self.state == 6:
                    img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)/255
                # print('draw right')
                img_draw = np.array(img_draw*255,np.uint8)
                img_draw = cv2.flip(img_draw,1)
                image = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                self.queue.put(image)
                self.event_generate('<<MessageGenerated>>')
                               
                is_call = True
                queue_ver.popleft()
                queue_ver_sparse.popleft()
                queue_frame.popleft()
            frame_i += 1
            if not is_call:
                # print('draw first')
                img_src = cv2.flip(img_src,1)
                image = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
                self.queue.put(image)
                self.event_generate('<<MessageGenerated>>')
                
                
            #######
            # image = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
            # self.queue.put(image)
            # self.event_generate('<<MessageGenerated>>')

    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            image = self.queue.get()
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image)
            self.view.configure(image=self.photo)


def main(args):
    root = tk.Tk()
    app = App(root, "Ella demo")
    root.mainloop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
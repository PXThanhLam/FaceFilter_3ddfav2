# coding: utf-8


from os import O_EXCL
import sys

from flask import Flask

sys.path.append('..')

import cv2
import numpy as np

from Sim3DR import RenderPipeline
from utils.functions import plot_image
from .tddfa_util import _to_ctype
from utils.uv import uv_tex,indices,process_uv, g_uv_coords
from Sim3DR import rasterize

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, tri, alpha=0.6, show_flag=False, wfp=None, with_bg_flag=True):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, overlap)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res


def render_filter(img, ver_lst, tri, alpha=0.1, filter=None,show_flag=False, wfp=None, with_bg_flag=True):
  
    filter = cv2.resize(filter,(256,256))
    # filter = cv2.cvtColor(filter, cv2.COLOR_BGR2RGB)
    uv_coords = process_uv(g_uv_coords.copy(), uv_h=256, uv_w=256) 
    uv_coords = np.array(uv_coords, dtype = np.uint8)[:,:2]
    uv_texture = filter[uv_coords[:,1],uv_coords[:,0],:]/255
    # cv2.imwrite('examples/filter.png',filter)
    overlap = np.zeros_like(img)

    for ver_  in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = rasterize(ver, tri, uv_texture, bg=overlap)

    if with_bg_flag:
        res = (img * 1.0 + 2.0*overlap)/255
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res

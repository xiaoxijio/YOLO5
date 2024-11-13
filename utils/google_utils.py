import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def attempt_download(weights):
    """查找本地是否有预训练模型"""
    weights = weights.strip().replace("'", '')
    file = Path(weights).name
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']  # available models

    if file in models and not os.path.isfile(weights):
        print('路径上没有yolov*.pt，自己去官网下')
        print('马上要报错喽！！！！！！！！！！')
        return

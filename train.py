import matplotlib.pyplot as plt
import shutil, os, math, time
from glob import glob
from tqdm import tqdm
import skimage.io as io
from skimage.transform import rescale, resize
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, metrics
import deepface
import mediapipe
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path", default="/workspace/data/cctv/cam2/reid/", type=str, help="media path")
    parser.add_argument("--params", default="/workspace/config/params.yaml", type=str, help="params path")
    parser.add_argument("--fps", default=30, type=int, help="decoding fps")
    parser.add_argument("--batch_size", default=30, type=int, help="batch size")
    parser.add_argument("--save_frame", action='store_true', help="save result frame")
    parser.add_argument("--save_video", action='store_true', help="save result video")

    option = parser.parse_known_args()[0]

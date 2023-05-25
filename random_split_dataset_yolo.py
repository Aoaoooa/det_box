#!/usr/bin/env python
import os
import os.path as osp
from glob import glob
import shutil
from random import shuffle

import cv2
from tqdm import tqdm

def move_one(img_p, src_label_d, img_dst_d, label_dst_d):
    label_p = osp.join(src_label_d, osp.basename(img_p)[:-3] + "txt")
    shutil.copy(img_p, img_dst_d)
    shutil.copy(label_p, label_dst_d)


def main(src_d, src_label_d, dst_d, file_ext="", split_list=[9,1]):

    if file_ext == "":
        src_img_p_s = glob(osp.join(src_d, "*"))
    else:
        src_img_p_s = glob(osp.join(src_d, f"*.{file_ext}"))

    shuffle(src_img_p_s)
    shuffle(src_img_p_s)
    shuffle(src_img_p_s)
    maxnum = len(src_img_p_s) * split_list[0] / 10
    print(maxnum)
    # mkdir
    train_d = osp.join(dst_d, "images/train")
    val_d = osp.join(dst_d, "images/val")

    train_label_d = osp.join(dst_d, "labels/train")
    val_label_d = osp.join(dst_d, "labels/val")


    os.makedirs(train_d, exist_ok=True)
    os.makedirs(val_d, exist_ok=True)
    os.makedirs(train_label_d, exist_ok=True)
    os.makedirs(val_label_d, exist_ok=True)

    for idx, img_p in enumerate(tqdm(src_img_p_s)):
        if idx < maxnum:
            move_one(img_p, src_label_d, train_d, train_label_d)
        else:
            move_one(img_p, src_label_d, val_d, val_label_d)


if __name__ == "__main__":
    
    src_d = "/home/ruis/datasets/lenovo_v2/train_barcode/all/images_0"
    src_label_d = "/home/ruis/datasets/lenovo_v2/train_barcode/all/yolo_labels"
    dst_d = "/home/ruis/datasets/lenovo_v2/train_barcode_yolo"

    main(src_d, src_label_d, dst_d)

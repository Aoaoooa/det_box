import os
import os.path as osp
import shutil
import json

import cv2
import numpy as np
from tqdm import tqdm
from vstools.img_tools.img_plt import xywh2xyxy, plot_one_box, plt_show_imgs, xywh2xyxy, xyxy2yololabel
from pycocotools.coco import COCO

def write_txt(dst, data):
    assert osp.splitext(dst)[-1] == '.txt', "仅限TXT文件"
    with open(dst, 'w') as f:
        f.writelines(data)
    return data

def main(json_p, out_d):
    os.makedirs(out_d, exist_ok=True)
    coco = COCO(json_p)
    img_ids = coco.getImgIds(catIds=[1])
    for img_id in tqdm(img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        bboxes = [i['bbox'] for i in anns]

        img_info = coco.loadImgs(img_id)
        img_name = img_info[0]["file_name"]
        img_p = osp.join(out_d, img_name)
        img_shape = (img_info[0]["height"], img_info[0]["width"])
        yolo_labels = [xyxy2yololabel(xywh2xyxy(bbox), img_shape) for bbox in bboxes]
        lines = ""
        for yolo_label in yolo_labels:
            lines += f"{int(anns[0]['category_id'] - 1)} {yolo_label[0]} {yolo_label[1]} {yolo_label[2]} {yolo_label[3]}\n"
        txt_fn = img_p.replace(".jpg", ".txt")
        write_txt(txt_fn, lines)




if __name__ == "__main__":
    json_p = "/home/chenm/data/yolov5_dataset/test/_annotations.coco.json"
    out_d = "/home/chenm/data/yolov5/test_labels"
    main(json_p, out_d)

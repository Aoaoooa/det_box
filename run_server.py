import argparse
import os
import platform
import sys
from pathlib import Path

import os.path as osp
os.environ['GREEN_BACKEND'] = 'gevent'

import eventlet
eventlet.monkey_patch()
import torch
import torch.backends.cudnn as cudnn

FILE = osp.abspath(osp.dirname(__file__))
MQSRV_ROOT = FILE + '/../../repo/mqsrv'
ROOT = FILE + '/../../repo/yolov5'
sys.path.insert(0, MQSRV_ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from kombu import Connection, Exchange
from mqsrv.base import get_rpc_exchange
from mqsrv.server import MessageQueueServer, run_server, make_server
from pydaemon import Daemon
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

class DetectionImage:
    rpc_prefix = 'detectionimage'

    def __init__(
                self,
                weights=ROOT / 'runs/train/exp5/weights/best.pt',  # model.pt path(s)
                source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                data=ROOT / 'data/my_box.yaml',  # dataset.yaml path
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_conf=False,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                nosave=False,  # do not save images/videos
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                update=False,  # update all models
                project='results',  # save results to project/name
                name='exp',  # save results to project/name
                exist_ok=False,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        check_requirements(exclude=('tensorboard', 'thop')) # 检测requirement.txt中的包是否安装好
        self.weights = weights
        self.source = source
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn

    def infer(self,img_path):
        if os.path.exists(img_path):
            self.source = str(img_path)
            save_img, webcam = self.source_judgment()
            save_dir = self.save_result()
            model, stride, names, pt = self.load_model()
            dataset, bs, vid_path, vid_writer = self.load_image(webcam, stride, pt)
            seen, dt, save_path = self.run(model, pt, bs, dataset, save_dir, webcam, save_img, vid_path, vid_writer, names)
            self.print_result(seen, save_img, save_dir, dt)
            return "结果路径:" + str(save_path)
        else:
            return "图片路径有误！"      

    def source_judgment(self):
        # source判断
        source = str(self.source) # 图片路径
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images 是否保存图片
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # Path(source).suffix[1:]=jpg IMG_FORMATS是图片格式 VID_FORMATS是视频格式
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # 判断source是否为网络流地址
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) # source.isnumeric() 判断是否打开电脑摄像头
        if is_url and is_file:
            self.source = check_file(source)  # download
        self.source = source # 保证是字符串类型
        return save_img, webcam

    def save_result(self):
        # Directories 新建保存结果文件夹
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        return save_dir

    def load_model(self):
        # Load model 加载模型权重
        self.device = select_device( ) # GPU or CPU ...
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half) # 加载模型
        stride, names, pt = model.stride, model.names, model.pt # 模型步长 目标名 是否为pytorch
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size 判断图片大小是否是步长的倍数
        return model, stride, names, pt

    def load_image(self, webcam, stride, pt):
        # Dataloader 加载待预测图片
        if webcam: 
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        return dataset, bs, vid_path, vid_writer
    
    def run(self, model, pt, bs, dataset, save_dir, webcam, save_img, vid_path, vid_writer, names):
        # Run inference 推理过程 检测
        model.warmup(imgsz=(1 if pt else bs, 3, *(self.imgsz)))  # warmup 热身
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0] # 存储结果信息
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim ， torch.Size([1,3,640,640])
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = model(im, augment=self.augment, visualize=visualize) #augment数据增强 visualize是否显示特征图 
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round() # 坐标映射

                    # Print results 
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results 是否保存图片结果
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}') #隐藏标签、置信度否
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            return seen, dt, save_path
    
    def print_result(self, seen, save_img, save_dir, dt):
         # Print results 打印出输出信息
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *(self.imgsz))}' % t)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)

if __name__ == "__main__":
    rpc_queue = 'server_rpc_queue_yolov5_m'
    server = make_server(
        rpc_routing_key=rpc_queue
    )
    detection = DetectionImage()
    server.register_rpc(detection.infer)
    run_server(server)
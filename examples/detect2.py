from PIL import Image,ImageDraw
from utils.operation import YOLO

def detect(onnx_path='/home/weih/repo/yolov5/runs/train/exp9/weights/best.onnx',img_path=r'/home/weih/repo/yolov5/datasets/data/images/train/0_jpg.rf.01fb6289e95e277a441458968dd8d7b7.jpg',show=True):
    '''
    检测目标，返回目标所在坐标如：
    {'crop': [57, 390, 207, 882], 'classes': 'person'},...]
    :param onnx_path:onnx模型路径
    :param img:检测用的图片
    :param show:是否展示
    :return:
    '''
    yolo = YOLO(onnx_path=onnx_path)
    det_obj = yolo.decect(img_path)

    # 结果
    print (det_obj)

    # 画框框
    if show:
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        for i in range(len(det_obj)):
            draw.rectangle(det_obj[i]['crop'],width=3)
        img.show()  # 展示


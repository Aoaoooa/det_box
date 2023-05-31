from PIL import Image,ImageDraw
import toml
import os.path as osp
FILE = osp.abspath(osp.dirname(__file__))
# from utils.operation import YOLO
from utils.operation import YOLO

class detect:
    #初始化
    def __init__(self,onnx_path,img_path,save_path):
        self.onnx_path = onnx_path
        self.img_path = img_path
        self.save_path = save_path

    #onnx推断
    def __infer__(self):
        self.yolo = YOLO(self.onnx_path)
        self.det_obj = self.yolo.decect(self.img_path)

    #绘图
    def draw(self):
        self.img = Image.open(self.img_path)
        draw = ImageDraw.Draw(self.img)
        for i in range(len(self.det_obj)):
            crop = self.det_obj[i]['crop']
            draw.rectangle(self.det_obj[i]['crop'],width=3)
            text_position = (crop[0],crop[1])
            text = "box"
            text_color = (255, 255, 0) 
            draw.text(text_position,text,text_color)
    
    #保存
    def save(self):

        # output_name = "output.jpg"
        save_path = self.save_path
        self.img.save(save_path)

    def main(img_path):
        config = toml.load("/home/weih/projects/det_box/config/config.toml")
        onnx_path = config["section1"]["onnx_path"]
        save_path = config["section1"]["save_path"]
        save_path = save_path + "box.jpg"
        # img_path = '../det_box/box1.jpg'
        detect_model = detect(FILE + onnx_path,FILE + img_path,FILE + save_path)
        detect_model.__infer__()
        detect_model.draw()
        detect_model.save()
        # print(img_path)

# img_path = '/../det_box/box1.jpg'
# main(img_path)
if __name__ == "__main__":
    detect.main(img_path)

    

        








# def detect(onnx_path='/home/weih/repo/yolov5/runs/train/exp9/weights/best.onnx',img_path='/home/weih/repo/yolov5/datasets/data/images/train/0_jpg.rf.01fb6289e95e277a441458968dd8d7b7.jpg',show=True):

#     yolo = YOLO(onnx_path=onnx_path)
#     det_obj = yolo.decect(img_path)

#     img = Image.open(img_path)
#     draw = ImageDraw.Draw(img)
#     for i in range(len(det_obj)):
#         crop = det_obj[i]['crop']
#         draw.rectangle(det_obj[i]['crop'],width=3)
#         text_position = (crop[0],crop[1])
#         text = "box"
#         text_color = (255, 255, 0) 
#         draw.text(text_position,text,text_color)

    # output_path = "output.jpg"
    # img.save(output_path)


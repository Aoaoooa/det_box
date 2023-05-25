import sys
import os.path as osp

FILE = osp.abspath(osp.dirname(__file__))
MQSRV_ROOT = FILE + '/../../repo/mqsrv'
sys.path.insert(0, MQSRV_ROOT)

from greenthread.monkey import monkey_patch; monkey_patch()
from greenthread.green import *
from loguru import logger
import traceback
import sys
from mqsrv.client import make_client

def main(broker_url):
    client = make_client()
    caller = client.get_caller('server_rpc_queue_yolov5_m')
    img_path = str(input("请输入需检测图片的绝对路径:"))
    exc, result = caller.detectionimage_infer(img_path)
    print(result)
    # 客户端释放
    client.release()

if __name__ == '__main__':
    main('pyamqp://')

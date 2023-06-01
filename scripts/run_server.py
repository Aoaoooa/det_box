# import sys
# import traceback
# import os
# import os.path as osp
# from pathlib import Path

# import eventlet
# eventlet.monkey_patch()

# cur_dir = osp.abspath(osp.dirname(__file__)) # 返回当前脚本文件所在目录路径的绝对路径
# sys.path.insert(0, cur_dir+'/../') # 将当前脚本文件的上一级目录路径插入到sys.path列表首位，以便导入其他模块时能正确搜索到上一级目录中的模块文件
# sys.path.insert(0,cur_dir+'/../../')
# # ROOT = str(cur_dir)+'/../../yolov5'
# # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# import os.path as osp
# FILE = osp.abspath(osp.dirname(__file__))

# # from yolov5.detect import *
# from kombu import Connection, Exchange
# from mqsrv. import get_rpc_exchange
# from mqsrv.server import MessageQueueServer, run_server, make_server
# from pydaemon import Daemon
# sys.path.append('../../../')
# from projects.det_box.examples.detect import *



import os
import sys

sys.path.append(".")
os.environ['GREEN_BACKEND'] = 'gevent'


# from det_box.detect import *
from det_box.detect import detect

from greenthread.monkey import monkey_patch; monkey_patch()
from mqsrv.server import make_server, run_server
from vstools.utils import toml2obj
from loguru import logger
import click


class Yolo:
    rpc_prefix = "yolo"
    def __init__(self) :
        # self.img_path = img_path
        pass

    def infer(self,img_path):
        # opt = detect.main(self.img_path)
        save_path = detect.main(img_path)
        return save_path
 

def echo(a):
    return a
def sum(a,b):
    return a + b
def fib_fn(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_fn(n - 1) + fib_fn(n - 2)

def handle_event(evt_type, evt_data):
    print ("handle event", evt_type, evt_data)

class FibClass:
    rpc_prefix = 'fibclass'

    def setup(self):
        print ("fib setuped")

    def teardown(self,evt_type, evt_data):
        print ("fib teared down")

    def fib(self, n):
        return fib_fn(n)

if __name__ == "__main__":

    fib_obj = FibClass()
    # addr = "amqp://guest:guest@0.0.0.0:5672/"
    rpc_queue = 'server_rpc_queue_wh'
    evt_queue = 'server_event_queue_wh '
    # 使用make_server()创建了一个名为"server"的服务器对象，用于创建一个与消息队列进行通信的服务器，rpc_routing_key（RPC路由键），event_routing_keys(事件路由键列表)
    server = make_server(
        # conn = addr,
        rpc_routing_key=rpc_queue,
        event_routing_keys=[evt_queue],
    )
#   这段代码展示了在一个服务器对象上注册了各种RPC(远程过程调用)和事件处理程序
    # server.register_rpc(echo)
    # server.register_rpc(fib_fn)
    # server.register_rpc(sum)
    # server.register_rpc(fib_obj.fib)
    yl = Yolo()
    server.register_rpc(yl.infer)
    # 这行代码注册了一个事件处理程序，它将处理名为”new"的事件。当服务器接收到名为"new"的事件时候，他将调用“handle_event”函数来处理事件。
    # server.register_event_handler('new', handle_event)
    server.register_event_handler('weihao',fib_obj.teardown)
    # 最终调用run_server函数来运行服务器，将服务器对象server作为参数传递给他。这将启动服务器并开始监听来自客户端的请求，并根据注册的RPC和事件处理程序来执行相应的逻辑。
    run_server(server)


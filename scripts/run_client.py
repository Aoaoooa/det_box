#!/usr/bin/env python
# import eventlet
# eventlet.monkey_patch()

import sys
import os.path as osp
cur_d = osp.dirname(__file__)
sys.path.insert(0, cur_d+'/../')

from greenthread.monkey import monkey_patch; monkey_patch()

from greenthread.green import *
from loguru import logger
import traceback
import sys
from mqsrv.client import make_client

def main(broker_url):
    # 这行代码创建了一个消息队列客户端对象，并将其赋值给变量client,根据make_client函数的定义，调用make_client()将使用默认的参数值来创建客户端对象。默认情况下，将使用默认的连接对象，默认的
    # RPC交换机，创建一个具有唯一名称的排他队列作为回调队列，并且使用默认的事件交换机。通过这行代码，你可以使用client变量来操作消息队列客户端，执行RPC调用、处理回调消息或发送事件消息，
    # 具体取决于MessageQueueClient类的实现和提供的方法。
    client = make_client()
    
    # 通过客户端对象client的方法获取了两个对象：caller和pubber
    caller = client.get_caller('server_rpc_queue_wh')
    pubber = client.get_pubber('server_event_queue_wh')

    # for i in range(10):
    # print ("sending echo")
    # 调用获取的caller对象的echo方法，并且传递了参数hello，根据代码中的赋值语句，返回值会被分配给exc和result两个变量。其中，exc是用于接收可能发生的异常
    # 信息的变量，而result是用于接收远程过程调用返回的结果的变量。
    # exc, result = caller.echo("hello")
    # print(result)
    # exc,result1 = caller.sum(2,3)
    # print(result1)

    # exc,result2 = caller.fibclass_fib(10)
    # print(result2)

    img_path = input("图片路径:")
    exc, result = caller.yolo_infer(img_path)
    print(result)
    # 这行代码调用了之前获取到的pubber对象
    pubber('new', {"hello":1})
    pubber('weihao', {"hello":7})
    client.release()

if __name__ == '__main__':
    main('pyamqp://')
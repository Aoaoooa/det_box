import os

from vstools.utils import toml2obj

from det_box.model import YoloOnnx


class Vison:
    def __init__(self, config_p):
        self.cfg = toml2obj(config_p)

        self.box_det_model = YoloOnnx(self.cfg.base.ckpt)

    
    def infer():
        

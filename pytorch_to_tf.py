import tempfile
import argparse
import sys
import time

# sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

from models.experimental import attempt_load
from models.common import Conv
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import os
import torch
import argparse
import onnx
from onnx_tf.backend import prepare
from onnxsim import simplify
import tensorflow as tf  
tf.config.experimental.enable_tensor_float_32_execution(False)
def main():
    onnx_save_path = 'all.onnx'

    model_path = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/runs/train/exp267/weights/71.pt"
    model_path = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/best_74_0.955/weights/74.pt"
    image_shape = 256
    model = attempt_load(model_path, map_location=torch.device('cpu'))
    model.eval()
    half = False
    input_sample = torch.randn((1, 3, image_shape, image_shape)) 

    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    # print(model.model[-1])
    # return
    y = model(input_sample)
    model.fuse()
    torch.onnx.export(model, input_sample, onnx_save_path, verbose=False, opset_version=12, input_names=['data'],
                      output_names=['stride_' + str(int(x)) for x in model.stride])

    # model.to_onnx(onnx_save_path, input_sample, do_constant_folding=True, export_params=True, opset_version=11)
    model = onnx.load(onnx_save_path)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.checker.check_model(model)
    input_shape = [1, 3, image_shape, image_shape]
    model_onnx, check = simplify(model, input_shapes={"data" :input_shape}, dynamic_input_shape=True)
    onnx.save(model, onnx_save_path)
    # onnx.save(model_onnx, save_path)
    assert check, "Simplified ONNX model could not be validated"
    PB_SAVE_PATH = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1_dynamic_shape_best955"
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf_rep = prepare(model_onnx) #, training_mode=True)#, gen_tensor_dict=True)  # prepare tf representation
    print(tf_rep.tensor_dict)
    print(tf_rep.inputs)
    print(tf_rep.outputs)
    tf_rep.export_graph(PB_SAVE_PATH) 
if __name__ == '__main__':
    main()
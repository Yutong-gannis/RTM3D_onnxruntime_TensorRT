import torch
import onnx
from onnxsim import simplify
import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
from lib.msra_resnet import get_pose_net


def create_model(heads, head_conv):
    model = get_pose_net(num_layers=18, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                state_dict[k] = model_state_dict[k]
    for k in model_state_dict:
        if not (k in state_dict):
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


class RTMDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda')
        self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.num_classes = 8  # 检测类别数
        self.heads = {'hm': self.num_classes, 'hps': 18, 'rot': 8, 'dim': 3, 'prob': 1}  # 检测头
        self.head_conv = 64
        self.model = create_model(self.heads, self.head_conv)
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device).eval()
        self.onnx_path = model_path[:-4] + ".onnx"
        self.onnx_half_path = model_path[:-4] + "_fp16.onnx"

    def convert(self, images):
        with torch.no_grad():
            torch.onnx.export(self.model, images,
                              self.onnx_path,
                              verbose=True,
                              input_names=['input'],
                              output_names=["hm", "hps", "rot", "dim", "prob"])
            print("Export ONNX successful. Model is saved at", self.onnx_path)

            onnx_model = onnx.load(self.onnx_path)  # load onnx model
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, self.onnx_path)

            onnx_model = onnxmltools.utils.load_model(self.onnx_path)
            onnx_model = convert_float_to_float16(onnx_model)
            onnxmltools.utils.save_model(onnx_model, self.onnx_half_path)
            print("Half model is saved at", self.onnx_half_path)


model_path = r"weights/kitti/model_250.pth"
images = torch.ones((1, 3, 384, 1280)).cuda()
detector = RTMDetector(model_path)
detector.convert(images)


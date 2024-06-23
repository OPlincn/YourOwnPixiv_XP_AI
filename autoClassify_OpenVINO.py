import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat

class YOLOV8Cls:
    def __init__(self, config):
        # 初始化模型配置参数
        self.inp_width = config['inpWidth']
        self.inp_height = config['inpHeight']
        self.onnx_path = config['onnx_path']
        self.initial_model()

    def initial_model(self):
        # 加载OpenVINO模型
        core = Core()
        model = core.read_model(self.onnx_path)
        ppp = PrePostProcessor(model)

        # 设置模型输入配置
        ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.RGB)
        ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.BGR).scale([255, 255, 255])
        ppp.input().model().set_layout(Layout("NCHW"))

        # 设置模型输出配置
        ppp.output().tensor().set_element_type(Type.f32)

        # 构建并编译模型
        model = ppp.build()
        self.compiled_model = core.compile_model(model, "CPU")
        self.infer_request = self.compiled_model.create_infer_request()

    def preprocess_img_letterbox(self, frame, new_shape=(224, 224), color=(114, 114, 114)): # color=(114, 114, 114) (255, 255, 255)
        shape = frame.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old) to fit the new shape
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # resize image and add padding
        if shape[::-1] != new_unpad:  # resize
            frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return frame

    def classify(self, frame):
        # 对输入图像进行预处理
        letterbox_frame = self.preprocess_img_letterbox(frame, (self.inp_width, self.inp_height))
        self.infer_request.infer({self.compiled_model.input().get_any_name(): np.expand_dims(letterbox_frame, 0)})

        # 获取模型输出
        output_tensor = self.infer_request.get_output_tensor()
        probs = output_tensor.data[0]
        return probs

def classify_images(input_folder, output_folder, detector, class_names):
    for class_name in class_names:
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

    for img_name in tqdm(os.listdir(input_folder), desc="Classifying images"):
        img_path = os.path.join(input_folder, img_name)
        if not os.path.isfile(img_path):
            continue

        frame = cv2.imread(img_path)
        probs = detector.classify(frame)
        class_id = np.argmax(probs)
        class_name = class_names[class_id]

        dest_folder = os.path.join(output_folder, class_name)
        shutil.copy(img_path, os.path.join(dest_folder, img_name))

if __name__ == "__main__":
    config = {
        'inpWidth': 224,
        'inpHeight': 224,
        'onnx_path': '/Volumes/dir01/Swap/pixivDataset/outModel/train14/weights/best_openvino_model/best.xml'  # 替换为模型路径
    }

    class_names = ["Manga", "Normal", "Sex"]

    detector = YOLOV8Cls(config)
    input_folder = '/Users/oplin/OpDocuments/VscodeProjects/PythonLessonWorks/PixivCrawler/download_images/2024_6_10__16'  # 替换为你的输入文件夹路径
    output_folder = '/Users/oplin/OpDocuments/VscodeProjects/PythonLessonWorks/PixivCrawler/download_images/testImgOut_train17'  # 替换为你的输出文件夹路径

    classify_images(input_folder, output_folder, detector, class_names)
    print("All images processed and copied to corresponding folders.")
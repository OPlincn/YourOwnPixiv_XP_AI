from eigen_cam import EigenCAM
from utils.image import show_cam_on_image, scale_cam_image
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def letterbox(image, new_shape=(224, 224), color=(114, 114, 114), auto=True, scale_fill=False, scaleup=True):
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return image, (r, r), (dw, dh)

model = YOLO("models/train14.pt",task="classify")
target_layers =[model.model.model[-3],model.model.model[-2]]
# target_layers =[model.model.model[-2]]
img = cv2.imread('/Users/oplin/OpDocuments/VscodeProjects/PythonLessonWorks/PixivCrawler/download_images/2024_6_10__16/119367533_p0.jpg') # 读入你的图片
# img = cv2.resize(img, (224, 224))
img, _, _ = letterbox(img, (224,224))
rgb_img = img.copy()
img = np.float32(img) / 255
cam = EigenCAM(model, target_layers,task='cls')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
print(img.shape)
plt.imshow(cam_image)
plt.show()

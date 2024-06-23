from ultralytics import YOLO
import cv2
class_names = ["Manga", "Normal", "Sex"]
model = YOLO("train11_ep20.pt")
result = model("Your Img or folder",show=True)
print(class_names[result[0].probs.top1])
cv2.waitKey(0)
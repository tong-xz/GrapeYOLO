from ultralytics import YOLO
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Detector(object):
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.cls_dict =  {0:'grape_cluster',1: 'index', 2:'measure_label', 3:'rachis_cluster'}

    def inference(self, image_path):
        return self.model([image_path])


    def get_all_boxes(self, results):
        '''
        @:param
        results : inference results of images

        @:return
        box_dict: {'class_name': box_tensor}
        '''
        box_dict = {}
        for result in results:
            boxes = result.boxes
            tensor_classes = boxes.cls

            tensor_classes = tensor_classes.cpu().numpy()

            boxes = boxes.xyxy.cpu().numpy()

            for index, tensor in zip(tensor_classes, boxes):
                box_dict[self.cls_dict[index]] = tensor
            # print(tensor_classes)
            #
            # box_dict = {}
            # for index, tensor in enumerate(boxes):
            #     box_dict[self.cls_dict[index]] = tensor
        return box_dict

    def plt_all_boxes(self, image_path, box_dict):
        '''
        :param image_path:
        :param box_dict:
        plot all the boxes of the image with all the detected objects
        '''
        image = Image.open(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image)
        plt.figure(figsize=(40, 40))
        # 在图片上绘制每个边界框
        for obj in box_dict.keys():
            box = box_dict[obj]
            # 创建一个矩形
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            # 添加矩形到绘图
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'{obj}', color='red', fontsize=10, verticalalignment='top')
        plt.show()



if __name__ == '__main__':

    image_path = '../13A.jpg'
    detector = Detector('../runs/detect/train3/weights/best.pt')
    result = detector.inference(image_path)
    box_dict = detector.get_all_boxes(result)

    detector.plt_all_boxes(image_path, box_dict)

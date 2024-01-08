from tools.ObjectDetector import Detector
import matplotlib.pyplot as plt
import numpy as np
from tools.Segmentator import Segmentator
import cv2
import os
import time

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_anns(anns):
    if len(anns) == 0:
        return

    # Convert the tensors to numpy arrays and sort them
    np_anns = [ann.numpy() for ann in anns]
    sorted_anns = sorted(np_anns, key=lambda x: np.sum(x), reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Initialize an image with transparency
    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:, :, 3] = 0

    i = 0
    for ann in sorted_anns:
        # Create a random color mask with transparency
        color_mask = np.concatenate([np.random.random(3), [0.35]])

        # Apply the mask to the annotation areas
        img[ann.astype(bool)] = color_mask

        # add text
        text = str(i)
        ys, xs = np.where(ann)
        centroid = (np.mean(xs), np.mean(ys))
        ax.scatter(centroid[0], centroid[1], color='green', marker='.', s=50, edgecolor='white', linewidth=1.25)
        # ax.text(centroid[0], centroid[1], text, color='white', ha='center', va='center')
        # i += 1

    ax.imshow(img)

if __name__ == "__main__":
    dir_path = "./samples/"
    duration_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            start = time.time()
            image_path = os.path.join(dir_path, file)


            # image_path = './samples/20A_back_grape.jpg'
            detector = Detector('../runs/detect/train3/weights/best.pt')
            result = detector.inference(image_path)
            box_dict = detector.get_all_boxes(result)
            detector.plt_all_boxes(image_path, box_dict)

            GrapeSegmenter = Segmentator(model_path='./FastSAM/weights/FastSAM-x.pt', DEVICE='cpu')
            prompt_process = GrapeSegmenter.start(image_path)
            ann_everything = GrapeSegmenter.seg_everything(prompt_process)
            grape_ann = GrapeSegmenter.filter_within_bbox(ann_everything, box_dict["grape_cluster"])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_box = np.array(box_dict["grape_cluster"])
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_anns(grape_ann)
            show_box(input_box, plt.gca())
            plt.axis('off')
            plt.title("{f}, num={n}".format(f=image_path, n=len(grape_ann)))
            # plt.savefig("./outputs/{f}_output.png".format(f=image_path), dpi=1080)
            plt.show()


            end = time.time()
            duration = end - start
            duration_list.append(float(duration))
            print("Total time usage: {duration}".format(duration=duration))


    print(duration_list)
    average = sum(duration_list) / len(duration_list)
    print("平均值:", average)







# import os
# os.chdir('../FastSAM')

from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np



class Segmentator:
    def __init__(self, model_path, DEVICE):
        self.model =FastSAM(model_path)
        self.DEVICE = DEVICE

    def start(self, IMAGE_PATH):
        everything_results = self.model(IMAGE_PATH, device=self.DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=self.DEVICE)
        return prompt_process

    def seg_everything(self, prompt_process):
        ann_everything = prompt_process.everything_prompt()
        return ann_everything

    def seg_with_bbox(self, prompt_process, bbox):
        ann_box = prompt_process.box_prompt(bbox=bbox)
        return ann_box

    def filter_within_bbox(self, anns, bbox):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        new_mask = []
        for ann in anns:
            # 寻找mask质点
            # 找到mask中所有为1的点的坐标, 如果没有找到点，返回None
            points = torch.nonzero(ann == 1)
            if points.nelement() == 0:
                return None
            ys, xs = np.where(ann)
            centroid = [np.mean(xs), np.mean(ys)]

            if (centroid[0] >= x1 and centroid[0] <= x2) and (centroid[1] >= y1 and centroid[1] <= y2) and int(
                    torch.sum(ann).item()) < 100000:
                new_mask.append(ann)
        return new_mask

    def plot_graph(self, prompt_process, ann, output_path):
        prompt_process.plot(annotations=ann, output_path=output_path)




if __name__ == '__main__':
    image_path = '../13A.jpg'
    GrapeSegmenter = Segmentator(model_path='../FastSAM/weights/FastSAM-x.pt', DEVICE='cpu')
    prompt_process = GrapeSegmenter.start(image_path)

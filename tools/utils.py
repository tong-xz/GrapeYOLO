import cv2
import numpy as np


def cal_measure_coefficient(image_path , xyxy):
    '''
    use the measure label to calculate the measure coefficient
    fixed length of blue label is 10cm
    :param image_path:
    :param xyxy: bbox in the form of xyxy
    :return: coefficient (cm/pixel)
    '''
    x, y = int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2)
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 应用阈值化突出目标
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最接近质心的轮廓
    closest_contour = None
    min_dist = float('inf')
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_contour = contour

    # 检查是否找到了最接近的轮廓
    if closest_contour is not None:


        # 获取最小包围矩形
        rect = cv2.minAreaRect(closest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 10)

        # 计算矩形的宽度和高度
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])

        return 10/height

if __name__ == '__main__':
    print(cal_measure_coefficient('../samples/1A_back_grape.jpg', [     318.09,      536.22,       555.9,      1896.2]))
# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import os
import sys
import pathlib
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

# project = 'DBNet.pytorch'  # 工作项目根目录
# sys.path.append(os.getcwd().split(project)[0] + project)
import time
import cv2
import torch

from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.3,type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    overlap_ratio = interArea / float(boxAArea + 1e-5)
    return iou, overlap_ratio

def draw_rects_clustered_by_row(img_path, txt_path, color=(0, 255, 0), thickness=2):
    assert os.path.exists(img_path), f"Không tìm thấy ảnh: {img_path}"
    assert os.path.exists(txt_path), f"Không tìm thấy file txt: {txt_path}"

    img = cv2.imread(img_path)
    boxes = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split(',')))
        coords = np.array(parts[:-1], dtype=np.int32).reshape(-1, 2)
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        boxes.append((x_min, y_min, x_max, y_max))

    # Remove small boxes overlapping large boxes
    def iou(a, b):
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        if area_a == 0 or area_b == 0:
            return 0
        return inter_area / min(area_a, area_b)

    filtered_boxes = []
    for i, box in enumerate(boxes):
        keep = True
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if iou(box, other) > 0.6:
                area_box = (box[2] - box[0]) * (box[3] - box[1])
                area_other = (other[2] - other[0]) * (other[3] - other[1])
                if area_box < 0.3 * area_other:
                    keep = False
                    break
        if keep:
            filtered_boxes.append(box)

    # Gom các box theo dòng
    lines = []
    heights = [y_max - y_min for _, y_min, _, y_max in filtered_boxes]
    avg_height = np.mean(heights)
    row_thresh = avg_height * 0.7

    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = box
        added = False
        for line in lines:
            ref_y = np.mean([b[1] for b in line])
            if abs(y_min - ref_y) < row_thresh:
                line.append(box)
                added = True
                break
        if not added:
            lines.append([box])

    # Sort các dòng theo y, sau đó trái -> phải
    lines.sort(key=lambda line: np.mean([b[1] for b in line]))
    number = 1
    for line in lines:
        line.sort(key=lambda b: b[0])

        # Chèn box vào khoảng trống quá lớn giữa các box cùng dòng
        new_line = []
        for i in range(len(line) - 1):
            box1 = line[i]
            box2 = line[i + 1]
            new_line.append(box1)
            gap = box2[0] - box1[2]
            if gap > avg_height * 0.8:
                # chèn box nghi ngờ
                y_min = min(box1[1], box2[1])
                y_max = max(box1[3], box2[3])
                x_min = box1[2] + gap // 4
                x_max = box2[0] - gap // 4
                if x_max > x_min:  # tránh box âm
                    new_line.append((x_min, y_min, x_max, y_max))
        new_line.append(line[-1])

        # Vẽ
        for box in new_line:
            x_min, y_min, x_max, y_max = box
            height = y_max - y_min
            expand = int(height * 0.5 / 2)
            y_min = max(0, y_min - expand)
            y_max = y_max + expand * 0.5
            x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
            cv2.putText(img, str(number), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            number += 1

    return img

if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    # 初始化网络
    model = Pytorch_model(args.model_path, post_p_thre=args.thre, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=args.polygon)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        # cv2.imwrite(output_path, img[:, :, ::-1])

        # Save initial output image (with drawn boxes from prediction)
        cv2.imwrite(output_path, img[:, :, ::-1])

        # Then overlay boxes from saved .txt (round-trip check)
        overlay_img = draw_rects_clustered_by_row(img_path, output_path.replace('_result.jpg', '.txt'))
        overlay_path = os.path.join(args.output_folder, img_path.stem + '_overlay.jpg')
        cv2.imwrite(overlay_path, overlay_img)

        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)

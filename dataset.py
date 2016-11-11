import os
import numpy as np
import tensorflow as tf
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

from utils.coco.coco import *

coco_num_class = 81

coco_class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush', 80: 'background'}

coco_class_colors = [[225, 239, 163], [202, 196, 172], [252, 182, 134], [170, 148, 215], [216, 243, 246], [229, 150,  89], [223, 226, 140], [154, 159, 166], [ 89, 146, 182], [199, 250, 161], [113, 233, 109], [135, 232,  89], [138, 216, 217], [ 87, 205, 191], [201, 106, 135], [158, 198, 159], [169, 147, 118], [187,  85, 107], [156,  97,  93], [176,  93, 108], [214, 190, 200], [212, 173, 198], [195, 188, 100], [162, 189, 192], [250, 122, 240], [122, 249, 106], [ 96, 110,  87], [230, 177, 203], [250, 201,  81], [195, 220, 198], [ 82, 143,  88], [ 96,  95, 105], [243, 153, 221], [153, 127,  81], [143, 211, 223], [188,  96, 250], [236, 233, 151], [185, 131, 198], [202, 232, 165], [188, 101, 213], [175, 184, 238], [223, 218, 245], [136, 210, 213], [156, 248,  85], [ 93, 221, 116], [200, 253,  91], [130, 210, 103], [210, 102, 212], [180, 178, 197], [160, 115, 138], [186, 229, 120], [184, 107,  86], [117, 229, 229], [186,  96, 139], [183, 215, 253], [106,  86, 154], [159, 184, 236], [217, 217, 194], [171, 108, 147], [ 94, 118, 231], [144, 242, 113], [183, 149, 230], [ 82,  98, 113], [166, 214, 170], [234, 128, 112], [166, 118, 178], [206, 138, 163], [239, 233, 178], [127, 238, 193], [180, 107, 208], [233, 230, 203], [ 92, 177, 113], [167, 209, 190], [245, 233, 109], [159,  92, 246], [208, 235, 166], [240,  91, 230], [118, 192, 103], [216, 102, 147], [170, 162, 200], [206, 252, 204]]

coco_class_to_category = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90, 80: 100}

coco_category_to_class = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 100: 80}


pascal_num_class = 21

pascal_class_names = {0: 'person', 1: 'bird', 2: 'cat', 3: 'cow', 4: 'dog', 5: 'horse', 6: 'sheep', 7: 'aeroplane', 8: 'bicycle', 9: 'boat', 10: 'bus', 11: 'car', 12: 'motorbike', 13: 'train', 14: 'bottle', 15: 'chair', 16: 'diningtable', 17: 'pottedplant', 18: 'sofa', 19: 'tvmonitor', 20: 'background'}

pascal_class_ids = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19, 'background': 20} 

pascal_class_colors = [[174, 220, 192], [116, 108, 127], [118, 144, 153], [189, 149, 122], [191,  93, 101], [154, 190, 115], [216, 148, 110], [230, 141, 249], [191, 217, 206], [156, 111, 135], [138, 147, 168], [138, 241, 227], [171, 113, 234], [139, 208, 147], [123, 205, 243], [145, 116, 119], [206, 204, 195], [157, 174, 227], [194, 205, 238], [183, 184, 164], [152, 248, 224]]


class DataSet():
    def __init__(self, img_ids, img_files, img_heights, img_widths, batch_size=1, anchor_files=None, gt_classes=None, gt_bboxes=None, is_train=False, shuffle=False):
        self.img_ids = np.array(img_ids)
        self.img_files = np.array(img_files)
        self.img_heights = np.array(img_heights)
        self.img_widths = np.array(img_widths)
        self.anchor_files = np.array(anchor_files)
        self.batch_size = batch_size
        self.gt_classes = gt_classes
        self.gt_bboxes = gt_bboxes
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.current_index = 0
        self.count = len(self.img_files)
        self.indices = list(range(self.count))
        self.num_batches = int(self.count/self.batch_size)
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()
        start, end = self.current_index, self.current_index + self.batch_size
        current_indices = self.indices[start:end]
        img_files = self.img_files[current_indices]
        if self.is_train:       
            anchor_files = self.anchor_files[current_indices]
            self.current_index += self.batch_size
            return img_files, anchor_files
        else:
            self.current_index += self.batch_size
            return img_files

    def has_next_batch(self):
        """ Determine whether there is any batch left. """
        return self.current_index + self.batch_size <= self.count


def prepare_train_coco_data(args):
    """ Prepare relevant COCO data for training the model. """
    image_dir, annotation_file, data_dir = args.train_coco_image_dir, args.train_coco_annotation_file, args.train_coco_data_dir
    batch_size = args.batch_size
    basic_model = args.basic_model
    num_roi = args.num_roi

    coco = COCO(annotation_file)

    img_ids = list(coco.imgToAnns.keys())
    img_files = []
    img_heights = []
    img_widths = []
    anchor_files = []
    gt_classes = []
    gt_bboxes = []

    for img_id in img_ids:
        img_files.append(os.path.join(image_dir, coco.imgs[img_id]['file_name'])) 
        img_heights.append(coco.imgs[img_id]['height']) 
        img_widths.append(coco.imgs[img_id]['width']) 
        anchor_files.append(os.path.join(data_dir, os.path.splitext(coco.imgs[img_id]['file_name'])[0]+'_'+basic_model+'_anchor.npz')) 

        classes = [] 
        bboxes = [] 
        for ann in coco.imgToAnns[img_id]: 
            classes.append(coco_category_to_class[ann['category_id']]) 
            bboxes.append([ann['bbox'][1], ann['bbox'][0], ann['bbox'][3]+1, ann['bbox'][2]+1]) 

        gt_classes.append(classes)  
        gt_bboxes.append(bboxes) 
 
    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, gt_classes, gt_bboxes, True, True)
    print("Dataset built.")
    return coco, dataset


def prepare_train_pascal_data(args):
    """ Prepare relevant PASCAL data for training the model. """
    image_dir, annotation_dir, data_dir = args.train_pascal_image_dir, args.train_pascal_annotation_dir, args.train_pascal_data_dir
    batch_size = args.batch_size
    basic_model = args.basic_model
    num_roi = args.num_roi

    files = os.listdir(annotation_dir)
    img_ids = list(range(len(files)))

    img_files = []
    img_heights = []
    img_widths = []
    anchor_files = []
    gt_classes = []
    gt_bboxes = []

    for f in files:
        annotation = os.path.join(annotation_dir, f)

        tree = ET.parse(annotation)
        root = tree.getroot()

        img_name = root.find('filename').text 
        img_file = os.path.join(image_dir, img_name)
        img_files.append(img_file) 

        img_id_str = os.path.splitext(img_name)[0]

        size = root.find('size')
        img_height = int(size.find('height').text)
        img_width = int(size.find('width').text)
        img_heights.append(img_height) 
        img_widths.append(img_width) 

        anchor_files.append(os.path.join(data_dir, img_id_str+'_'+basic_model+'_anchor.npz')) 

        classes = [] 
        bboxes = [] 
        for obj in root.findall('object'): 
            class_name = obj.find('name').text
            class_id = pascal_class_ids[class_name]
            classes.append(class_id) 

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bboxes.append([ymin, xmin, ymax-ymin+1, xmax-xmin+1]) 

        gt_classes.append(classes)  
        gt_bboxes.append(bboxes) 
 
    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, gt_classes, gt_bboxes, True, True)
    print("Dataset built.")
    return dataset


def prepare_val_coco_data(args):
    """ Prepare relevant COCO data for validating the model. """
    image_dir, annotation_file = args.val_coco_image_dir, args.val_coco_annotation_file

    coco = COCO(annotation_file)

    img_ids = list(coco.imgToAnns.keys())
    img_files = []
    img_heights = []
    img_widths = []

    for img_id in img_ids:
        img_files.append(os.path.join(image_dir, coco.imgs[img_id]['file_name']))
        img_heights.append(coco.imgs[img_id]['height'])         
        img_widths.append(coco.imgs[img_id]['width'])         

    print("Building the validation dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return coco, dataset


def prepare_val_pascal_data(args):
    """ Prepare relevant PASCAL data for validating the model. """
    image_dir, annotation_dir = args.val_pascal_image_dir, args.val_pascal_annotation_dir

    files = os.listdir(annotation_dir)
    img_ids = list(range(len(files)))

    img_files = []
    img_heights = []
    img_widths = []

    pascal = {}

    for f in files:
        annotation = os.path.join(annotation_dir, f)

        tree = ET.parse(annotation)
        root = tree.getroot()

        img_name = root.find('filename').text 
        pascal[img_name] = []

        img_file = os.path.join(image_dir, img_name)
        img_files.append(img_file) 
 
        size = root.find('size')
        img_height = int(size.find('height').text)
        img_width = int(size.find('width').text)
        img_heights.append(img_height) 
        img_widths.append(img_width) 

        for obj in root.findall('object'): 
            class_name = obj.find('name').text
            class_id = pascal_class_ids[class_name]
            temp = obj.find('difficult')
            difficult = int(temp.text) if temp!=None else 0

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            pascal[img_name].append({'class_id': class_id, 'bbox':[xmin, ymin, xmax, ymax], 'difficult': difficult})

    print("Building the validation dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return pascal, dataset


def eval_pascal_one_class(pascal, detections, c):
    """ Evaluate the detection result for one class on PASCAL dataset. """
    gts = {} 
    num_objs = 0 
    for img_name in pascal:
        gts[img_name] = []
        for obj in pascal[img_name]:
            if obj['class_id'] == c and obj['difficult']==0:
                gts[img_name] += [{'bbox':obj['bbox'], 'detected': False}]
                num_objs += 1

    dts = []
    scores = []
    num_dets = 0
    for img_name in detections:
        for dt in detections[img_name]:
            if dt['class_id'] == c:
                dts.append([img_name, dt['bbox'], dt['score']])
                scores.append(dt['score'])
                num_dets += 1
    
    # Sort the detections based on their scores
    scores = np.array(scores, np.float32)
    sorted_idx = np.argsort(scores)[::-1]

    tp = np.zeros((num_dets))
    fp = np.zeros((num_dets))

    for i in tqdm(list(range(num_dets))):
        idx = sorted_idx[i]
        img_name = dts[idx][0]
        bbox = dts[idx][1]      
        gt_bboxes = np.array([obj['bbox'] for obj in gts[img_name]], np.float32)        

        # Compute the max IoU of current detection with the ground truths
        max_iou = 0.0
        if gt_bboxes.size > 0:
            ixmin = np.maximum(gt_bboxes[:, 0], bbox[0])
            iymin = np.maximum(gt_bboxes[:, 1], bbox[1])
            ixmax = np.minimum(gt_bboxes[:, 2], bbox[2])
            iymax = np.minimum(gt_bboxes[:, 3], bbox[3])

            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)

            area_intersect = iw * ih
            area_union = (bbox[2] - bbox[0] + 1.0) * (bbox[3] - bbox[1] + 1.0) + (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.0) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.0) - area_intersect

            ious = area_intersect / area_union
            max_iou = np.max(ious, axis=0)
            j = np.argmax(ious)

        # Determine if the current detection is a true or false positive
        if max_iou > 0.5:
            if not gts[img_name][j]['detected']:
                tp[i] = 1.0
                gts[img_name][j]['detected'] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    # Accumulate the numbers of true and false positives
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    # Compute the average precision based on these data
    rec = tp * 1.0 / num_objs
    prec = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    print('average precision for class %s = %f' %(pascal_class_names[c], ap))

    return ap


def eval_pascal(pascal, detections):
    """ Evaluate the detection result on PASCAL dataset. """
    ap = 0.0 
    for i in range(pascal_num_class-1):
        ap += eval_pascal_one_class(pascal, detections, i)
    ap = ap / (pascal_num_class-1)
    print('mean average precision = %f' %ap)
    return ap


def prepare_test_data(args):
    """ Prepare relevant data for testing the model. """
    image_dir = args.test_image_dir

    files = os.listdir(image_dir)
    files = [f for f in files if f.lower().endswith('.jpg')]

    img_ids = list(range(len(files)))
    img_files = []
    img_heights = []
    img_widths = []
      
    for f in files:
        img_path = os.path.join(image_dir, f)
        img_files.append(img_path)
        img = cv2.imread(img_path)
        img_heights.append(img.shape[0]) 
        img_widths.append(img.shape[1]) 

    print("Building the testing dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths)
    print("Dataset built.")
    return dataset


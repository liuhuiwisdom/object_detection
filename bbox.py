import time
import numpy as np

def iou_bbox(bboxes1, bboxes2):
    """ Compute the IoUs between bounding boxes. """
    bboxes1 = np.array(bboxes1, np.float32)
    bboxes2 = np.array(bboxes2, np.float32)
    
    intersection_min_y = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    intersection_max_y = np.minimum(bboxes1[:, 0] + bboxes1[:, 2] - 1, bboxes2[:, 0] + bboxes2[:, 2] - 1)
    intersection_height = np.maximum(intersection_max_y - intersection_min_y + 1, np.zeros_like(bboxes1[:, 0]))

    intersection_min_x = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    intersection_max_x = np.minimum(bboxes1[:, 1] + bboxes1[:, 3] - 1, bboxes2[:, 1] + bboxes2[:, 3] - 1)
    intersection_width = np.maximum(intersection_max_x - intersection_min_x + 1, np.zeros_like(bboxes1[:, 1]))

    area_intersection = intersection_height * intersection_width
    area_first = bboxes1[:, 2] * bboxes1[:, 3]
    area_second = bboxes2[:, 2] * bboxes2[:, 3]
    area_union = area_first + area_second - area_intersection
    
    iou = area_intersection * 1.0 / area_union
    iof = area_intersection * 1.0 / area_first
    ios = area_intersection * 1.0 / area_second

    return iou, iof, ios

def param_bbox(bboxes, anchors):
    """ Parameterize bounding boxes with respect to anchors. Namely, (y,x,h,w)->(ty,tx,th,tw). """
    bboxes = np.array(bboxes, np.float32)
    anchors = np.array(anchors, np.float32)

    tyx = (bboxes[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    thw = np.log(bboxes[:, 2:] / anchors[:, 2:])

    t = np.concatenate((tyx, thw), axis=1)
    return t

def unparam_bbox(t, anchors, max_shape=None):
    """ Unparameterize bounding boxes with respect to anchors. Namely, (ty,tx,th,tw)->(y,x,h,w). """
    t = np.array(t, np.float32)
    anchors = np.array(anchors, np.float32)

    yx = t[:, :2] * anchors[:, 2:] + anchors[:, :2]
    hw = np.exp(t[:, 2:]) * anchors[:, 2:]

    bboxes = np.concatenate((yx, hw), axis=1)

    if max_shape != None:
        bboxes = rectify_bbox(bboxes, max_shape)

    return bboxes

def rectify_bbox(bboxes, max_shape):
    """ Clip bounding boxes to image boundary if necessary. """ 
    bboxes = np.array(bboxes, np.int32)
    n = bboxes.shape[0]
    if n == 0:
        return bboxes

    h, w = max_shape

    bboxes[:, 0] = np.maximum(bboxes[:, 0], np.zeros((n)))
    bboxes[:, 0] = np.minimum(bboxes[:, 0], (h-1) * np.ones((n)))
    bboxes[:, 1] = np.maximum(bboxes[:, 1], np.zeros((n)))
    bboxes[:, 1] = np.minimum(bboxes[:, 1], (w-1) * np.ones((n)))
    bboxes[:, 2] = np.maximum(bboxes[:, 2], np.ones((n)))
    bboxes[:, 2] = np.minimum(bboxes[:, 2], h * np.ones((n)) - bboxes[:, 0])
    bboxes[:, 3] = np.maximum(bboxes[:, 3], np.ones((n)))
    bboxes[:, 3] = np.minimum(bboxes[:, 3], w * np.ones((n)) - bboxes[:, 1])

    return bboxes
    
def convert_bbox(bboxes, old_shape, new_shape):
    """ Map bounding boxes in old image shape to their counterparts in new image shape. """
    bboxes = np.array(bboxes, np.float32)
    if bboxes.shape[0] == 0:
        return bboxes

    oh, ow = old_shape
    nh, nw = new_shape

    bboxes[:, 0] = bboxes[:, 0] * nh / oh
    bboxes[:, 1] = bboxes[:, 1] * nw / ow
    bboxes[:, 2] = bboxes[:, 2] * nh / oh
    bboxes[:, 3] = bboxes[:, 3] * nw / ow

    bboxes = rectify_bbox(bboxes, new_shape)
    return bboxes

def expand_bbox(bboxes, max_shape, factor=1.5):
    """ Enlarge bounding boxes by the given factor (without changing their centers). """
    bboxes = np.array(bboxes, np.float32)
    n = bboxes.shape[0]
    if n == 0:
        return bboxes

    H = bboxes[:, 2] * factor
    W = bboxes[:, 3] * factor
    Y = bboxes[:, 0] - bboxes[:, 2] * (factor * 0.5 - 0.5)
    X = bboxes[:, 1] - bboxes[:, 3] * (factor * 0.5 - 0.5)

    Y = np.expand_dims(Y, 1)
    X = np.expand_dims(X, 1)
    H = np.expand_dims(H, 1)
    W = np.expand_dims(W, 1)
    expanded_bboxes = np.concatenate((Y,X,H,W), axis=1)

    expanded_bboxes = rectify_bbox(expanded_bboxes, max_shape)
    return expanded_bboxes

def generate_anchors(img_shape, feat_shape, scale, ratio, factor=1.5):
    """ Generate the anchors. """
    ih, iw = img_shape
    fh, fw = feat_shape
    n = fh * fw

    # Compute the coordinates of the anchors
    j = np.array(list(range(fh)))
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, fw))
    j = j.reshape((-1))

    i = np.array(list(range(fw)))
    i = np.expand_dims(i, 0)
    i = np.tile(i, (fh, 1))
    i = i.reshape((-1))

    s = np.ones((n)) * scale
    r0 = np.ones((n)) * ratio[0]
    r1 = np.ones((n)) * ratio[1]

    h = s * r0
    w = s * r1
    y = (j + 0.5) * ih / fh - h * 0.5
    x = (i + 0.5) * iw / fw - w * 0.5

    ph = h * factor
    pw = w * factor
    py = y - h * (factor * 0.5 - 0.5)
    px = x - w * (factor * 0.5 - 0.5)

    # Determine if the anchors cross the boundary
    anchor_is_untruncated = np.ones((n), np.int32)  
    anchor_is_untruncated[np.where(y<0)[0]] = 0
    anchor_is_untruncated[np.where(x<0)[0]] = 0
    anchor_is_untruncated[np.where(h+y>ih)[0]] = 0
    anchor_is_untruncated[np.where(w+x>iw)[0]] = 0

    parent_anchor_is_untruncated = np.ones((n), np.int32)  
    parent_anchor_is_untruncated[np.where(py<0)[0]] = 0
    parent_anchor_is_untruncated[np.where(px<0)[0]] = 0
    parent_anchor_is_untruncated[np.where(ph+py>ih)[0]] = 0
    parent_anchor_is_untruncated[np.where(pw+px>iw)[0]] = 0

    # Clip the anchors if necessary
    y = np.maximum(y, np.zeros((n)))
    x = np.maximum(x, np.zeros((n)))
    h = np.minimum(h, ih-y)
    w = np.minimum(w, iw-x)

    py = np.maximum(py, np.zeros((n)))
    px = np.maximum(px, np.zeros((n)))
    ph = np.minimum(ph, ih-py)
    pw = np.minimum(pw, iw-px)

    y = np.expand_dims(y, 1)
    x = np.expand_dims(x, 1)
    h = np.expand_dims(h, 1)
    w = np.expand_dims(w, 1)
    anchors = np.concatenate((y, x, h, w), axis=1)
    anchors = np.array(anchors, np.int32)

    py = np.expand_dims(py, 1)
    px = np.expand_dims(px, 1)
    ph = np.expand_dims(ph, 1)
    pw = np.expand_dims(pw, 1)
    parent_anchors = np.concatenate((py, px, ph, pw), axis=1)
    parent_anchors = np.array(parent_anchors, np.int32)

    # Count the number of untruncated anchors
    num_anchor = np.array([n], np.int32)
    num_untruncated_anchor = np.sum(anchor_is_untruncated)
    num_untruncated_anchor = np.array([num_untruncated_anchor], np.int32)
    num_untruncated_parent_anchor = np.sum(parent_anchor_is_untruncated)
    num_untruncated_parent_anchor = np.array([num_untruncated_parent_anchor], np.int32)

    return num_anchor, anchors, anchor_is_untruncated, num_untruncated_anchor, parent_anchors, parent_anchor_is_untruncated, num_untruncated_parent_anchor

def label_anchors(anchors, anchor_is_untruncated, gt_classes, gt_bboxes, background_id, iou_low_threshold=0.41, iou_high_threshold=0.61):
    """ Get the labels of the anchors. Each anchor can be labeled as positive (1), negative (0) or ambiguous (-1). Truncated anchors are always labeled as ambiguous. """
    n = anchors.shape[0]
    k = gt_bboxes.shape[0]
    
    # Compute the IoUs of the anchors and ground truth boxes
    tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
    tiled_gt_bboxes = np.tile(np.expand_dims(gt_bboxes, 0), (n, 1, 1))

    tiled_anchors = tiled_anchors.reshape((-1, 4))
    tiled_gt_bboxes = tiled_gt_bboxes.reshape((-1, 4))

    ious, ioas, iogs = iou_bbox(tiled_anchors, tiled_gt_bboxes)
    ious = ious.reshape(n, k)
    ioas = ioas.reshape(n, k)
    iogs = iogs.reshape(n, k)

    # Label each anchor based on its max IoU
    max_ious = np.max(ious, axis=1)
    max_ioas = np.max(ioas, axis=1)
    max_iogs = np.max(iogs, axis=1)
    
    best_gt_bbox_ids = np.argmax(ious, axis=1)

    labels = -np.ones((n), np.int32)
    positive_idx = np.where(max_ious >= iou_high_threshold)[0]
    negative_idx = np.where(max_ious < iou_low_threshold)[0]
    labels[positive_idx] = 1
    labels[negative_idx] = 0
   
    # Truncated anchors are always ambiguous
    ignore_idx = np.where(anchor_is_untruncated==0)[0]
    labels[ignore_idx] = -1

    bboxes = gt_bboxes[best_gt_bbox_ids]

    classes = gt_classes[best_gt_bbox_ids]
    classes[np.where(labels<1)[0]] = background_id

    max_ious[np.where(anchor_is_untruncated==0)[0]] = -1
    max_ioas[np.where(anchor_is_untruncated==0)[0]] = -1
    max_iogs[np.where(anchor_is_untruncated==0)[0]] = -1

    return labels, bboxes, classes, max_ious, max_ioas, max_iogs

def nms(scores, bboxes, k, iou_threshold=0.7, score_threshold=0.5):
    """ Non-maximum suppression. """
    n = len(scores)

    idx = np.argsort(scores)[::-1]
    sorted_scores = scores[idx]
    sorted_bboxes = bboxes[idx]
    
    top_k_ids = []
    size = 0
    i = 0

    while i < n and size < k:
        if sorted_scores[i] < score_threshold:
            break
        top_k_ids.append(i)
        size += 1
        i += 1
        while i < n:
            tiled_bbox_i = np.tile(sorted_bboxes[i], (size, 1)) 
            ious, iofs, ioss = iou_bbox(tiled_bbox_i, sorted_bboxes[top_k_ids])
            max_iou = np.max(ious)
            if max_iou > iou_threshold:
                i += 1
            else:
                break

    return size, sorted_scores[top_k_ids], sorted_bboxes[top_k_ids]


def postprocess(scores, classes, bboxes, iou_threshold=0.3, score_threshold=0.5):
    """ Post-process the detection results. Non-maximum suppression. """
    n = len(scores)
        
    det_num = 0
    det_classes = []    
    det_scores = []
    det_bboxes = []

    idx = np.argsort(scores)[::-1]
    sorted_scores = scores[idx]
    sorted_bboxes = bboxes[idx]
    sorted_classes = classes[idx]

    top_k_ids = []
    i = 0

    while i < n:
        if sorted_scores[i] < score_threshold:
            break

        top_k_ids.append(i)
        det_num += 1
        det_scores.append(sorted_scores[i])
        det_bboxes.append(sorted_bboxes[i])
        det_classes.append(sorted_classes[i])
        i += 1

        while i < n:
            tiled_bbox_i = np.tile(sorted_bboxes[i], (det_num, 1)) 
            flags = (sorted_classes[top_k_ids]==sorted_classes[i])*1.0 
            ious, iofs, ioss = iou_bbox(tiled_bbox_i, sorted_bboxes[top_k_ids]) 
            max_iou = np.max(ious) 
        #    max_iof = np.max(iofs*flags) 
        #    max_ios = np.max(ioss*flags) 
        #    temp = np.max((max_iof, max_ios))
            if max_iou > iou_threshold:
                i += 1
            else:
                break

    return det_num, np.array(det_scores, np.float32), np.array(det_classes, np.int32), np.array(det_bboxes, np.int32)


def postprocess2(scores, classes, bboxes, iou_threshold=0.2, score_threshold=0.5):
    """ Post-process the detection results. Non-maximum suppression within each class. """
    n = len(scores)
 
    count_per_class = {cls:0 for cls in classes}
    bbox_per_class = {cls:[] for cls in classes}
    score_per_class = {cls:[] for cls in classes}

    for i in range(n):
        count_per_class[classes[i]] += 1
        bbox_per_class[classes[i]] += [bboxes[i]]
        score_per_class[classes[i]] += [scores[i]]
        
    det_num = 0
    det_classes = []    
    det_scores = []
    det_bboxes = []

    for cls in count_per_class:
        current_count = count_per_class[cls]
        current_scores = np.array(score_per_class[cls], np.float32)
        current_bboxes = np.array(bbox_per_class[cls], np.int32)

        idx = np.argsort(current_scores)[::-1]
        sorted_scores = current_scores[idx]
        sorted_bboxes = current_bboxes[idx]

        top_k_ids = []
        size = 0
        i = 0

        while i < current_count:
            if sorted_scores[i] < score_threshold:
                break
            top_k_ids.append(i)
            det_num += 1
            det_classes.append(cls)
            det_scores.append(sorted_scores[i])
            det_bboxes.append(sorted_bboxes[i])
            size += 1
            i += 1

            while i < current_count:
                tiled_bbox_i = np.tile(sorted_bboxes[i], (size, 1))
                ious, iofs, ioss = iou_bbox(tiled_bbox_i, sorted_bboxes[top_k_ids])
                max_iou = np.max(ious)
            #    max_iof = np.max(iofs)
            #    max_ios = np.max(ioss)
            #    temp = np.max((max_iof, max_ios))
                if max_iou > iou_threshold:
                    i += 1
                else:
                    break

    return det_num, np.array(det_scores, np.float32), np.array(det_classes, np.int32), np.array(det_bboxes, np.int32)


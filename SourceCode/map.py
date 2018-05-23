"""
Reference: http://tcd.miovision.com/challenge/dataset/
"""

import numpy as np

f = open("res_test_resnet.csv", "r")
results = []
for line in f.readlines():
    results.append(line.strip().split(","))
r1 = np.array(results)

f = open("src/MIO-TCD-Localization/MIO-TCD-Localization/gt_test.csv", "r")
gt = []
for line in f.readlines():
    gt.append(line.strip().split(","))
g1 = np.array(gt)

def iou_ratio(bbox_1, bbox_2):
    '''
        Compute the IoU ratio between two bounding boxes
    '''

    bi = [max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
          min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3])]

    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    ov = 0

    if iw > 0 and ih > 0:
        ua = (bbox_1[2] - bbox_1[0] + 1) * (bbox_1[3] - bbox_1[1] + 1) +              (bbox_2[2] - bbox_2[0] + 1) * (bbox_2[3] - bbox_2[1] + 1) -              iw * ih
        ov = iw * ih / float(ua)

    return ov

def VOCap(rec, prec):
    '''
        Compute the average precision following the code in Pascal VOC toolkit
    '''
    mrec = np.array(rec).astype(np.float32)
    mprec = np.array(prec).astype(np.float32)
    mrec = np.insert(mrec, [0, mrec.shape[0]], [0.0, 1.0])
    mprec = np.insert(mprec, [0, mprec.shape[0]], [0.0, 0.0])

    for i in range(mprec.shape[0]-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])

    i = np.ndarray.flatten(np.array(np.where(mrec[1:] != mrec[0:-1]))) + 1
    ap = np.sum(np.dot(mrec[i] - mrec[i-1], mprec[i]))
    return ap

def compute_metric_class(gt, res, cls, minoverlap):
    '''
        Computes the localization metrics of a given class
            * precision-recall curve
            * average precision
    '''

    # loading the ground truth for class cls
    npos = 0
    gt_cls = {}
    for img in gt.keys():
        index = np.array(gt[img]['class']) == cls
        BB = np.array(gt[img]['bbox'])[index]
        det = np.zeros(np.sum(index[:]))
        npos += np.sum(index[:])
        gt_cls[img] = {'BB': BB,
                       'det': det}

    # loading the detection result
    score = np.array(res[cls]['score'])
    imgs = np.array(res[cls]['img'])
    BB = np.array(res[cls]['bbox'])

    # sort detections by decreasing confidence
    si = np.argsort(-score)
    
    imgs = imgs[si]
    
    if len(si) == 0:
        return (0, 0, 0)
    BB = BB[si, :]

    # assign detections to ground truth objects
    nd = len(score)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        img = imgs[d]

        bb = BB[d, :]
        ovmax = 0

        for j in range(len(gt_cls[img]['BB'])):
            bbgt = gt_cls[img]['BB'][j]
            ov = iou_ratio(bb, bbgt)
            if ov > ovmax:
                ovmax = ov
                jmax = j

        if ovmax >= minoverlap:
            if not gt_cls[img]['det'][jmax]:
                tp[d] = 1
                gt_cls[img]['det'][jmax] = 1
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp/(fp+tp)

    ap = VOCap(rec, prec)

    return rec, prec, ap

classes = ['articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle',
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian',
           'pickup_truck', 'single_unit_truck', 'work_van']

gtd = {}
for row in gt:
    img = row[0]
    cls = row[1]
    bbox = np.array(row[2:]).astype('float32')

    if img in gtd:
        gtd[img]['class'].append(cls)
        gtd[img]['bbox'].append(bbox)
    else:
        gtd[img] = {}
        gtd[img]['class'] = [cls]
        gtd[img]['bbox'] = [bbox]
        gtd[img][fixed_class] = False
    if cls == fixed_class:
        gtd[img][fixed_class] = True

res = {}
for cls in classes:
    res[cls] = {}
    res[cls]['img'] = []
    res[cls]['score'] = []
    res[cls]['bbox'] = []

for row in results:
    img, cls, score = row[0], row[1], float(row[2])
    bbox = np.array(row[3:]).astype('float32')

    # Compute the iou with all ground truth bounding boxes

    if gtd[img][fixed_class]:
        ovmax = 0
        label = cls
        for k, gt_bb in enumerate(gtd[img]['bbox']):
            ov = iou_ratio(bbox, gt_bb)
            if ov > ovmax:
                label = gtd[img]['class'][k]
                ovmax = ov

        if ovmax > 0.5:
            if label == fixed_class:
                cls = fixed_class

    res[cls]['img'].append(img)
    res[cls]['score'].append(score)
    res[cls]['bbox'].append(bbox)

metrics = {}
map = []
for cls in classes:
    #cls = fixed_class
    rec, prec, ap = compute_metric_class(gtd, res, cls, 0.5)
    metrics[cls] = {}
    metrics[cls]['recall'] = rec
    metrics[cls]['precision'] = prec
    metrics[cls]['ap'] = ap
    map.append(ap)

print(map)
metrics["map"] = np.mean(np.array(map))
print("Mean Average Precision:", metrics["map"])

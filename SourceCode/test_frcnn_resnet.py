"""
https://github.com/jinfagang/keras_frcnn
"""

# Importing libraries
import cv2
import numpy as np
import argparse
import os
from keras import backend as K
from keras.layers import Input
from keras.models import Model

# Importing functions from referenced code
from faster_rcnn.roi_helpers import rpn_to_roi, apply_regr, non_max_suppression_fast
import faster_rcnn.resnet as rn
from faster_rcnn.visualize import draw_boxes_and_label_on_image_cv2


from faster_rcnn.parser import parse_mapping

# Write bounding box results to CSV File for later calculating the MAP score
f = open("res_test_resnet.csv", "w")

def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return real_x1, real_y1, real_x2, real_y2

def predict_single_image(img_path, model_rpn, model_classifier_only, class_mapping):
    img = cv2.imread(img_path)
    if img is None:
        print('reading image failed.')
        exit(0)

    X, ratio = format_img(img)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    result = rpn_to_roi(Y1, Y2, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    bbox_threshold = 0.7

    # apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    for jk in range(result.shape[0] // 32 + 1):
        rois = np.expand_dims(result[32 * jk:32 * (jk + 1), :], axis=0)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // 32:
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], 32, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded

        [p_cls, p_regr] = model_classifier_only.predict([F, rois])

        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(p_cls[0, ii, :])
            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            (x, y, w, h) = rois[0, ii, :]
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= 8.0
                ty /= 8.0
                tw /= 4.0
                th /= 4.0
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            boxes[cls_num].append(
                [16 * x, 16 * y, 16 * (x + w), 16 * (y + h),
                 np.max(p_cls[0, ii, :])])
    
    # add some nms to reduce many boxes
    for cls_num, box in boxes.items():
        boxes_nms = non_max_suppression_fast(box, overlap_thresh=0.5)
        boxes[cls_num] = boxes_nms
        #print(class_mapping[cls_num] + ":")
        for b in boxes_nms:
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
            f.write(",".join([img_path.split("/")[-1].split(".")[0], class_mapping[cls_num], str(b[-1]), str(b[0]), str(b[1]), str(b[2]), str(b[3])]) + "\n")

    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    result_path = './resnet_aug_results_images/{}.jpg'.format(os.path.basename(img_path).split('.')[0])
    cv2.imwrite(result_path, img)

def predict(args_):
    path = args_.path
    model = "model_trained/frcnn_resnet_aug.hdf5"
    class_mapping = parse_mapping(path)
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(32, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = rn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = 3 * 3 # 3 for number of scales and 3 number of aspect ratios
    rpn_layers = rn.rpn(shared_layers, num_anchors)
    classifier = rn.classifier(feature_map_input, roi_input, 32, nb_classes=len(class_mapping), trainable=True)
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print("Loading weights from", model)
    #model_rpn.load_weights(cfg.model_path_rpn, by_name=True)
    #model_classifier.load_weights(cfg.model_path_class, by_name=True)
    
    model_rpn.load_weights(model, by_name=True)
    model_classifier.load_weights(model, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            predict_single_image(os.path.join(path, img_name), model_rpn, model_classifier_only, class_mapping)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier_only, class_mapping)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='images/00000000.jpg', help='image path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    predict(args)

"""
Reference: https://github.com/jinfagang/keras_frcnn
"""

#Importing libraries
from random import shuffle, choice
import time
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

#Importing functions from referenced code
from faster_rcnn.data_generators import get_anchor_gt
from faster_rcnn import losses as losses_fn
from faster_rcnn.roi_helpers import rpn_to_roi, calc_iou
from faster_rcnn import resnet as rn


from faster_rcnn.parser import parse_data, parse_mapping

def train():

    num_rois = 32
    epochs = 250
    model = "model_trained/frcnn_resnet.hdf5"
    ground_truth = "MIO-TCD-Localization/gt_train.csv"
    pretrained_model = "model/resnet50_weights_tf_dim_ordering_tf_kernels.h5"

    # load filenames and class mappings
    images, label_dict = parse_data(ground_truth)
    map1 = parse_mapping(ground_truth)

    # shuffle the data
    shuffle(images)

    # create train and validation data split
    train_data = []
    val_data = []

    # randomly choosing train and test
    for image in images:
        if np.random.randint(0, 6) == 0:
            val_data.append(image)
        else:
            train_data.append(image)

    print("Num train samples", len(train_data))
    print("Num val samples", len(val_data))

    data_gen_train = get_anchor_gt(train_data, label_dict, rn.get_img_output_length, K.image_dim_ordering(), mode='train')
    data_gen_val = get_anchor_gt(val_data, label_dict, rn.get_img_output_length, K.image_dim_ordering(), mode='val')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network
    shared_layers = rn.nn_base(img_input, trainable=True)

    # define the RPN using the base layers
    num_anchors = 3 * 3 # 3 for number of scales and 3 number of aspect ratios
    rpn = rn.rpn(shared_layers, num_anchors)

    classifier = rn.classifier(shared_layers, roi_input, num_rois, nb_classes=len(label_dict), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this model holds both the RPN and the classifier
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    print("Loading pretrained weights from", pretrained_model)
    model_rpn.load_weights(pretrained_model, by_name=True)
    model_classifier.load_weights(pretrained_model, by_name=True)

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(label_dict) - 1)], metrics={'dense_class_{}'.format(len(label_dict)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 250
    losses = np.zeros((epoch_length, 5))
    best_loss = np.inf
    iteration = 0

    for epoch in range(epochs):
        print("Epoch ", epoch + 1)

        # Training on Train Dataset

        X, Y, img_data = next(data_gen_train)
        P_rpn = model_rpn.predict_on_batch(X)
        result = rpn_to_roi(P_rpn[0], P_rpn[1], K.image_dim_ordering(), True, 0.7, 300)
        X2, Y1, Y2, IouS = calc_iou(result, img_data, map1)

        pos_samples = np.where(Y1[0, :, -1] == 0)
        if len(pos_samples) == 0:
            pos_samples = []
        else:
            pos_samples = pos_samples[0]

        neg_samples = np.where(Y1[0, :, -1] == 1)
        if len(neg_samples) == 0:
            neg_samples = []
        else:
            neg_samples = neg_samples[0]

        if num_rois > 1:
            if len(pos_samples) < num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, num_rois // 2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=True).tolist()

            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            if np.random.randint(0, 2) == 0:
                sel_samples = choice(pos_samples)
            else:
                sel_samples = choice(neg_samples)

        model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

        # Testing on Validation Dataset

        X, Y, img_data = next(data_gen_val)
        P_rpn = model_rpn.predict_on_batch(X)
        result = rpn_to_roi(P_rpn[0], P_rpn[1], K.image_dim_ordering(), True, 0.7, 300)
        X2, Y1, Y2, IouS = calc_iou(result, img_data, map1)

        pos_samples = np.where(Y1[0, :, -1] == 0)
        if len(pos_samples) == 0:
            pos_samples = []
        else:
            pos_samples = pos_samples[0]

        neg_samples = np.where(Y1[0, :, -1] == 1)
        if len(neg_samples) == 0:
            neg_samples = []
        else:
            neg_samples = neg_samples[0]

        if num_rois > 1:
            if len(pos_samples) < num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, num_rois // 2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, num_rois - len(selected_pos_samples), replace=True).tolist()

            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            if np.random.randint(0, 2):
                sel_samples = choice(neg_samples)
            else:
                sel_samples = choice(pos_samples)

        loss_class = model_classifier.test_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

        losses[iteration, 2] = loss_class[1]
        losses[iteration, 3] = loss_class[2]
        losses[iteration, 4] = loss_class[3]

        if iteration == epoch_length * (epoch + 1):
            rpn_cls_loss = np.mean(losses[:, 0])
            rpn_regr_loss = np.mean(losses[:, 1])
            class_cls_loss = np.mean(losses[:, 2])
            class_regr_loss = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            print("Classifier accuracy for bounding boxes from RPN:", class_acc)
            print("Loss RPN classifier:", rpn_cls_loss)
            print("Loss RPN regression:", rpn_regr_loss)
            print("Loss Detector classifier:", class_cls_loss)
            print("Loss Detector regression:", class_regr_loss)

            total_loss = rpn_cls_loss + rpn_regr_loss + class_cls_loss + class_regr_loss

            if total_loss < best_loss:
                best_loss = total_loss
                model_all.save_weights(model)

        iteration += 1

if __name__ == '__main__':
    train()
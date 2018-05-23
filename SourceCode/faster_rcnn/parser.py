import cv2
import numpy as np

def parse_mapping(ground_truth):
    map = {}
    f = open(ground_truth, "r")
    for line in f.readlines():
        label = line.strip().split(",")[1]
        if label not in map:
            map[label] = len(map)

def parse_data(ground_truth):
    images = {}

    label_dict = {}
    f = open(ground_truth, "r")
    for line in f.readlines():
        line_split = line.strip().split(",")
        (fname, label, x1, y1, x2, y2) = line_split

        # create dictionary of labels
        if label in label_dict:
            label_dict[label] += 1
        else:
            label_dict[label] = 1

        fname = "MIO-TCD-Localization/train/" + fname + ".jpg"

        if fname not in images:
            images[fname] = {}
            img = cv2.imread(fname)
            h, w, _ = img.shape

            # for every new image
            images[fname]["filepath"] = fname
            images[fname]["height"] = h
            images[fname]["width"] = w
            images[fname]["bboxes"] = []

        images[fname]['bboxes'].append(
            {
                "x1" : int(float(x1)),
                "y1" : int(float(y1)),
                "x2" : int(float(x2)),
                "y2" : int(float(y2)),
                "class" : label
            })

    list1 = []
    for image in images:
        list1.append(images[image])

    return list1, label_dict


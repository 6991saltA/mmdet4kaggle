import matplotlib.pyplot as plt
import json
import sys
import random
# from pylab import xticks, yticks, np


def readJson(jsonPath):
    bbox_mAP = []
    with open(jsonPath, 'r') as f:
        for line in f.readlines():
            json_data = json.loads(line)
            if 'bbox_mAP' in json_data:
                bbox_mAP.append(json_data['bbox_mAP'])
            continue
        f.close()
        return bbox_mAP


def findMax(numList, maxNum):
    for i in range(len(numList)):
        if numList[i] == maxNum:
            return i
        i = i + 1


def plot(x, y, lineStyle):
    plt.xlabel('Epoch')
    plt.ylabel('bbox_mAP_50')
    plt.plot(x, y, lineStyle)


if __name__=='__main__':
    path1 = r'D:\Atlas\md\work_dirs\faster_rcnn_resswin_noPretrained\20230318_115753.log.json'  # ssd
    # path2 = r'D:\Atlas\mmdetection\work_dirs\faster_rcnn_swin-t_p4-w7_fpn_1x_coco\swin.json'  # ssd
    # path3 = r'D:\Atlas\mmdetection\work_dirs\faster_rcnn_r50_fpn_1x_coco\resnet.json'

    x = []
    for i in range(48):
        x.append(i)

    bbox_mAP_combined = readJson(path1)
    bbox_mAP_combined.insert(0, 0)
    print(max(bbox_mAP_combined))
    # bbox_mAP_swin = readJson(path2)
    # bbox_mAP_swin.insert(0, 0)

    # bbox_mAP_resnet = readJson(path3)
    # bbox_mAP_resnet.insert(0, 0)

    plot(x, bbox_mAP_combined, 'b-')
    # plot(x, bbox_mAP_swin, 'g-')
    # plot(x, bbox_mAP_resnet, 'r-')

    plt.legend(['Parallel'])
    plt.show()


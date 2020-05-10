import time
import logging
import cv2
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--classes', type=str, default='./data/kitti.names',
                    help='类别配置路径')
parser.add_argument('--weights', type=str, default='./checkpoints/yolov3_train_1.tf',
                    help='训练好的模型位置')
parser.add_argument('--tiny', type=bool, default=False, help='加载的模型类型yolov3 or yolov3-tiny')
parser.add_argument('--size', type=int, default=416,
                    help='图片大小')
parser.add_argument('--img', type=str, default='./data/kitti/data_object_image_2/testing/image_2/000008.png',
                    help='输入预测图片的位置')
parser.add_argument('--output', type=str, default='./output.jpg',
                    help='输出图片结果的位置')
parser.add_argument('--num_classes', type=int, default=6,
                    help='总共类别数量')


def main(args):
    # 1、初始化模型并加载权重
    if args.tiny:
        yolo = YoloV3Tiny(classes=args.num_classes)
    else:
        yolo = YoloV3(classes=args.num_classes)

    yolo.load_weights(args.weights)
    logging.info('加载模型权重weights')

    # 加载目标类型
    class_names = [c.strip() for c in open(args.classes).readlines()]

    # 2、加载图片处理图片并使用模型进行预测
    img = tf.image.decode_image(open(args.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images  (img, args.size)

    # 记录时间
    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('耗时: {}'.format(t2 - t1))

    logging.info('检测结果:')
    print(boxes, scores, classes, nums)
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))
    # 3、显示图片并将图片框画出
    img = cv2.imread(args.image)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(args.output, img)
    logging.info('output saved to: {}'.format(args.output))


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)

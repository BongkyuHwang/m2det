# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [32., 64., 118.4, 172.8, 227.2, 281.6],
    'max_sizes': [64, 118.4, 172.8, 227.2, 281.6, 336.],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 128, 320],
    'min_sizes': [22.4, 48., 105.6, 163.2, 220.8, 278.4],
    'max_sizes': [48., 105.6, 163.2, 220.8, 278.4, 336.],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

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
    'milestones': [100, 150, 200, 250, 300],
    'max_epoch': 300,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'gammas' : [0.1, 0.5, 0.1, 0.1, 0.1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 107, 320],
    'min_sizes': [25.6, 48., 105.6, 163.2, 220.8, 278.4],
    'max_sizes': [48., 105.6, 163.2, 220.8, 278.4, 336.],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 81,
    'milestones': [90, 110, 130, 150, 160],
    'max_epoch': 160,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'gammas' : [0.1, 0.5, 0.1, 0.1, 0.1],
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 107, 320],
    'min_sizes': [25.6, 48., 105.6, 163.2, 220.8, 278.4],
    'max_sizes': [48., 105.6, 163.2, 220.8, 278.4, 336.],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

st = {
    'num_classes': 2,
    'milestones': [90, 110, 130, 150, 160],
    'max_epoch': 160,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'gammas' : [0.1, 0.5, 0.1, 0.1, 0.1],
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 107, 320],
    'min_sizes': [25.6, 48., 105.6, 163.2, 220.8, 278.4],
    'max_sizes': [48., 105.6, 163.2, 220.8, 278.4, 336.],
    'aspect_ratios': [[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SynthText',
}

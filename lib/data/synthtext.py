from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from layers import box_utils

ST_ROOT = osp.join(HOME, 'data/SynthText/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
ST_CLASSES = ('background', 'text')

class STAnnotationTransform(object):
    """Transforms a SynthText annotation into a Tensor of bbox coords and label index
    """
    def __call__(self, polygons):
        """
        Args:
            polygons (numpy array): SynthText text annotation polygons as a numpy array
        Returns:
            a numpy array containing of bounding boxes  [bbox coords, class idx]
        """
        bboxes = box_utils.polygon_to_bbox(polygons)
        rboxes = box_utils.polygon_to_rbox(polygons)
        return np.hstack([bboxes, polygons, rboxes])

class STDetection(data.Dataset):
    """`SynthText Detection dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, transform=None,
                 target_transform=STAnnotationTransform(), dataset_name='SynthText'):
        self.root = root
        self.gt_path = os.path.join(self.root, "gt.pt")

        data = torch.load(self.gt_path)
        
        self.image_names = data["image_names"]
        self.polygons = data["polygons"]
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.image_names)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:Tuple
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """

        path = osp.join(self.root, self.image_names[index])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, _ = img.shape
        target = self.polygons[index]
        if self.transform is not None:
            #target = np.array(target)
            img, boxes, labels = self.transform(img, target[:,:8], target[:, 8])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        path = osp.join(self.root, self.image_names[index])
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        return self.polygons[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
   
    

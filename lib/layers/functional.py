from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
from torch.autograd import Function
from .box_utils import decode, nms
from data import voc as cfg

def focal_loss(input, target, num_classes, alpha=0.25, gamma=2.):
    target = torch.eye(num_classes)[target].to(target.device)
    sigmoid_p = input.sigmoid()
    zeros = sigmoid_p.new(sigmoid_p.shape).fill_(0)
    pos_p_sub = torch.where(target > zeros, target - sigmoid_p, zeros)
    pos_n_sub = torch.where(target == zeros, sigmoid_p, zeros)
    loss = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-10, 1.0)) - (1 - alpha) * (pos_n_sub ** gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-10, 1.0))

    return loss.sum()

def prior_box(cfg):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    image_size = cfg['min_dim']
    # number of priors for feature map location (either 4 or 6)
    num_priors = len(cfg['aspect_ratios'])
    variance = cfg['variance'] or [0.1]
    feature_maps = cfg['feature_maps']
    min_sizes = cfg['min_sizes']
    max_sizes = cfg['max_sizes']
    steps = cfg['steps']
    aspect_ratios = cfg['aspect_ratios']
    clip = cfg['clip']
    version = cfg['name']
    for v in variance:
        if v <= 0:
            raise ValueError('Variances must be greater than 0')

    mean = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f), repeat=2):
            f_k = image_size / steps[k]
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = min_sizes[k]/image_size
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(s_k * (max_sizes[k]/image_size))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in aspect_ratios[k]:
                mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
    # back to torch land
    output = torch.Tensor(mean).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)

    return output

def detect(num_classes, bkg_label, top_k, conf_thresh, nms_thresh, loc_data, conf_data, prior_data):
    """
    At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    if nms_thresh <= 0:
        raise ValueError('nms_threshold must be non negative.')
    variance = cfg['variance']

    num = loc_data.size(0)  # batch size
    num_priors = prior_data.size(0)
    output = torch.zeros(num, num_classes, top_k, 5)
    conf_preds = conf_data.view(num, num_priors,
                                    num_classes).transpose(2, 1)

    # Decode predictions into bboxes.
    for i in range(num):
        decoded_boxes = decode(loc_data[i], prior_data, variance)
        # For each class, perform nms
        conf_scores = conf_preds[i].clone()

        for cl in range(1, num_classes):
            c_mask = conf_scores[cl].gt(conf_thresh)
            scores = conf_scores[cl][c_mask]
            if scores.dim() == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            # idx of highest scoring and non-overlapping boxes per class
            ids, count = nms(boxes, scores, nms_thresh, top_k)
            output[i, cl, :count] = \
                torch.cat((scores[ids[:count]].unsqueeze(1),
                            boxes[ids[:count]]), 1)
    flt = output.contiguous().view(num, -1, 5)
    _, idx = flt[:, :, 0].sort(1, descending=True)
    _, rank = idx.sort(1)
    flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
    return output

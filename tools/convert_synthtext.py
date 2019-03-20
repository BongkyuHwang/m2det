import torch
import numpy as np 
import scipy.io
import os 
import struct

from get_image_size import get_image_size

def convert(data_path, max_slope=None, polygon=False): 
    gt_path = os.path.join(data_path, 'gt.mat') 
    classes = ['Background', 'Text'] 
    print('loding ground truth data...') 
    data = scipy.io.loadmat(gt_path) 
    num_samples = data['imnames'].shape[1] 
    print('processing samples...') 
    image_names = [] 
    bboxes = [] 
    texts = [] 
    for image_name, text, boxes in zip(data["imnames"][0], data["txt"][0], data["wordBB"][0]):
        image_name = image_name[0]
        text = [s.strip().split() for s in text] 
        text = [w for s in text for w in s] 
        # read image size only when size changes 
        if image_name.split('_')[-1].split('.')[0] == '0': 
            img_width, img_height = get_image_size(os.path.join(data_path, image_name)) 
        
        # data['wordBB'][0,i] has shape (x + y, points, words) = (2, 4, n) 
        if len(boxes.shape) == 2: 
            boxes = boxes[:,:,None] 
        boxes = boxes.transpose(2,1,0) 
        boxes[:,:,0] /= img_width 
        boxes[:,:,1] /= img_height 
        boxes = boxes.reshape(boxes.shape[0],-1) 
        # fix some bugs in the SynthText dataset 
        eps = 1e-3 
        p1, p2, p3, p4 = boxes[:,0:2], boxes[:,2:4], boxes[:,4:6],boxes[:,6:8] 
            
        # fix twisted boxes (897 boxes, 0.012344 %) 
        if True: 
            mask = np.linalg.norm(p1 + p2 - p3 - p4, axis=1) < eps 
            boxes[mask] = np.concatenate([p1[mask], p3[mask], p2[mask], p4[mask]], axis=1) 
            
        # filter out bad boxes (528 boxes, 0.007266 %) 
        if True: 
            mask = np.ones(len(boxes), dtype=np.bool) 
            # filter boxes with zero width (173 boxes, 0.002381 %) 
            boxes_w = np.linalg.norm(p1-p2, axis=1) 
            boxes_h = np.linalg.norm(p2-p3, axis=1) 
            mask = np.logical_and(mask, boxes_w > eps) 
            mask = np.logical_and(mask, boxes_h > eps) 
            # filter boxes that are too large (62 boxes, 0.000853 %) 
            mask = np.logical_and(mask, np.all(boxes > -1, axis=1)) 
            mask = np.logical_and(mask, np.all(boxes < 2, axis=1)) 
            # filter boxes with all vertices outside the image (232 boxes, 0.003196 %) 
            boxes_x = boxes[:,0::2] 
            boxes_y = boxes[:,1::2] 
            mask = np.logical_and(mask, 
                    np.sum(np.logical_or(np.logical_or(boxes_x < 0, boxes_x > 1), 
                            np.logical_or(boxes_y < 0, boxes_y > 1)), axis=1) < 4) 
            # filter boxes with center outside the image (336 boxes, 0.004624 %) 
            boxes_x_mean = np.mean(boxes[:,0::2], axis=1) 
            boxes_y_mean = np.mean(boxes[:,1::2], axis=1) 
            mask = np.logical_and(mask, np.logical_and(boxes_x_mean > 0, boxes_x_mean < 1)) 
            mask = np.logical_and(mask, np.logical_and(boxes_y_mean > 0, boxes_y_mean < 1)) 
            boxes = boxes[mask] 
            text = np.asarray(text)[mask] 
                
        # only boxes with slope below max_slope 
        if not max_slope == None: 
            angles = np.arctan(np.divide(boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1])) 
            angles[angles < 0] += np.pi 
            angles = angles/np.pi*180-90 
            boxes = boxes[np.abs(angles) < max_slope] 
            
        # only images with boxes 
        if len(boxes) == 0: 
            continue 
            
        if not polygon: 
            xmax = np.max(boxes[:,0::2], axis=1) 
            xmin = np.min(boxes[:,0::2], axis=1) 
            ymax = np.max(boxes[:,1::2], axis=1) 
            ymin = np.min(boxes[:,1::2], axis=1) 
            boxes = np.array([xmin, ymin, xmax, ymax]).T 
                
        # append classes 
        boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1) 
        image_names.append(image_name) 
        bboxes.append(boxes) 
        texts.append(text)

    return image_names, bboxes, texts

if __name__ == "__main__":
    image_names, bboxes, texts = convert("/home/mcmas/data/SynthText", None, True)
    meta = dict()
    meta["image_names"] = image_names
    meta["polygons"] = bboxes
    meta["texts"] = texts
    torch.save(meta, "/home/mcmas/data/SynthText/gt.pt")

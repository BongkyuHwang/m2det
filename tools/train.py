import torch
import numpy
import time

import _init_paths
import data
import layers
import models
import utils

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(0)

def train():
    device = "cuda" if torch.cuda.is_available() == True else "cpu"
    print(device)
    net = models.M2Det(num_classes=data.coco["num_classes"], model_name="se_resnext101_32x4d")
    net.to(device)
    print(data.COCO_ROOT)
    dataset = data.COCODetection(root="/home/mcmas/data/coco2017", image_set='train2017', transform=utils.augmentations.SSDAugmentation(data.coco["min_dim"], net.settings["mean"], net.settings["std"]))
    #dataset = data.COCODetection(root=data.COCO_ROOT, transform=augmentations.SSDAugmentation(data.coco["min_dim"], net.settings["mean"], net.settings["std"]))
    data_loader = torch.utils.data.DataLoader(dataset, 6, num_workers=8, shuffle=True, collate_fn=data.detection_collate, pin_memory=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=2e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 30, 40], gamma=0.1)
    criterion = layers.MultiBoxLoss(num_classes=data.coco["num_classes"], overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False, use_gpu=True)

    for epoch in range(1, 51):
        loc_loss = 0
        conf_loss = 0
        scheduler.step()
        for itr, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    t0 = time.time()
                    out = net(images)
                    optimizer.zero_grad()
                    loss_l, loss_c = criterion(out, targets)
                    loss = loss_l + loss_c
                    loss.backward()
                    optimizer.step()
                    t1 = time.time()
                    loc_loss += loss_l.item()
                    conf_loss += loss_c.item()
                    if itr % 10 == 0:
                        print("timer : %.4f sec." %(t1 - t0))
                        print("iter " +repr(itr)+ " || loss : %.4f || " % (loss.item()), end=" ")
        print("saving state, epoch : ", epoch)
        torch.save(net.state_dict(), "m2det320_coco_" + repr(epoch) + ".pth")
    torch.save(net.state_dict(), "m2det320_coco_finish.pth")


train()

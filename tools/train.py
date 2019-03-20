import torch
import numpy
import time
import argparse
import os
import datetime

import _init_paths
import data
import layers
import models
import utils
import optim
parser = argparse.ArgumentParser(description="M2Det Training")
parser.add_argument("--dataset", default="VOC", choices=["VOC", "COCO", "ST"], 
                    type=str, help="VOC|COCO|ST")
parser.add_argument("--basenet", default="se_resnext101_32x4d", 
                    choices=["pnasnet5large", "se_resnext101_32x4d"], 
                    type=str, help="Base Pretrained model")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size for trainig")

args = parser.parse_args()

torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
numpy.random.seed(0)
def set_parameter_requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag

def train():
    device = "cuda" if torch.cuda.is_available() == True else "cpu"
    print(device)

    if args.dataset == "VOC":
        cfg = data.voc
        dataset_root = data.VOC_ROOT
        dataset_cls = data.VOCDetection
    elif args.dataset == "COCO":
        cfg = data.coco
        dataset_root = data.COCO_ROOT
        dataset_cls = data.COCODetection
    elif args.dataset == "ST":
        cfg = data.st
        dataset_root = data.ST_ROOT
        dataset_cls = data.STDetection
        milestones = []
    
    if not os.path.exists("models"):
        os.mkdir("models")
    root_path = os.path.join("models", "%s_%s"%(args.dataset, datetime.datetime.now()))
    os.mkdir(root_path)

    net = models.M2Det(num_classes=cfg["num_classes"], model_name=args.basenet)
    net.train()
    dataset = dataset_cls(root=dataset_root, 
                        transform=utils.augmentations.SSDAugmentation(cfg["min_dim"], 
                        net.settings["mean"], net.settings["std"]))
    net = torch.nn.DataParallel(net)
    set_parameter_requires_grad(net.module.fe, False)
 
    net.to(device)
    print(len(dataset)) 
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=8, shuffle=True, collate_fn=data.detection_collate, pin_memory=True, drop_last=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=4e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg["milestones"], cfg["gammas"])
    criterion = layers.MultiBoxLoss(num_classes=cfg["num_classes"], overlap_thresh=0.5, prior_for_matching=True, bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5, encode_target=False, use_gpu=True)


    for epoch in range(1, cfg["max_epoch"]+1):
        loc_loss = 0
        conf_loss = 0
        if epoch == 6:
            set_parameter_requires_grad(net.module.fe, True)

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
                    if itr % 10 == 0 and itr != 0:
                        print("timer : %.4f sec." %(t1 - t0))
                        print("epoch " + repr(epoch) +" || iter " +repr(itr)+ " || loss : %.4f || " % (loss.item()) + "loc_loss : %.4f ||" %(loss_l.item()) + " conf_loss : %.4f || "%(loss_c.item()), end=" ")
        with open(os.path.join(root_path, "loss.log"), "a") as lp:
            lp.write("%d %.4f %.4f %.4f\n"%(int(epoch), (loc_loss + conf_loss) / itr, loc_loss / itr, conf_loss / itr))
        if epoch != 0 and epoch % 10 == 0:
            print("saving state, epoch : ", epoch)
            torch.save(net.state_dict(), os.path.join(root_path, "m2det320_%s_%03d.pth"%(args.dataset, int(epoch))))

    torch.save(net.state_dict(), os.path.join(root_path, "m2det320_%s_final.pth"%(args.dataset)))

train()

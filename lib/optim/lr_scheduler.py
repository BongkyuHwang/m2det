import torch
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones shoud be a list of"
                            " increasing integers. Got {}", milestones)
        if isinstance(gammas, list) and len(milestones) != len(gammas):
            raise ValueError("Gammas shoud be a list having same lenth with milestones")

        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = bisect_right(self.milestones, self.last_epoch)
        return [base_lr * self.gammas[idx-1] ** idx for base_lr in self.base_lrs]

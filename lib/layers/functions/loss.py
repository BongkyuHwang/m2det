import torch

def focal_loss(input, target, num_classes, alpha=0.25, gamma=2.):
    t = torch.eye(num_classes)[target].to(target.device)
    #t = t[:, 1:]
    print(t.shape, input.shape)
    xt = input * (2 * t - 1)
    pt = (2 * xt + 1).sigmoid()

    w = alpha * t + (1 - alpha) * (1 - t)
    loss = -w * pt.log() / gamma
    return loss.sum()
    



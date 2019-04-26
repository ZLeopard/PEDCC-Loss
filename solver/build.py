# encoding: utf-8

import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        # if "bias" in key:
        #     lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        #     weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    lr_schduler = getattr(torch.optim.lr_scheduler, cfg.SOLVER.LR_SCHDULER)(optimizer, cfg.SOLVER.MILE_STONES, gamma=0.1, last_epoch=-1)
    return optimizer, lr_schduler

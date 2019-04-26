# encoding: utf-8

import logging

import time
import torch
import os
import sys
from tqdm import tqdm
sys.path.append('.')
from config.defaults import _C as cfg
from utils.utils import read_pkl
from utils.utils import get_acc
from utils.utils import get_lfw_list
from utils.utils import get_lfw_acc


def do_train(
        cfg,
        model,
        metric_fc,
        train_loader,
        val_loader,
        optimizer,
        lr_schedule,
        loss_fn,
        loss_fn2,
        logger,
):
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    lfw_test_list = cfg.LFW_TEST_LIST

    # device_ids = [0, 1, 2, 3]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    if metric_fc is not None:
        metric_fc.to(device)
    map_dict = read_pkl()
    for epoch in range(epochs):
        lr_schedule.step()
        model.train()

        # zero the loss
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        label_mse_tensor = torch.tensor([])
        for iter, (images, targets) in enumerate(tqdm(train_loader)):
            iter += 1
            images = images.to(device)
            targets = targets.to(device)

            # read PEDCC weights

            tensor_empty = torch.Tensor([]).to(device)
            for target_index in targets:
                tensor_empty = torch.cat((tensor_empty, map_dict[target_index.item()].float().to(device)), 0)

            label_mse_tensor = tensor_empty.view(-1, 512)
            label_mse_tensor = label_mse_tensor.to(device)   # PEDCC of each class

            # forward
            output_ = model(images)
            output = output_[0]
            t_loss1 = loss_fn(output, targets)     # PEDCC-AMSOFTMAX
            t_loss2 = loss_fn2(output_[1], label_mse_tensor)
            t_loss2 = t_loss2**cfg.METRIC.N
            t_loss = t_loss1 + t_loss2

            # backward
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            train_loss += t_loss.item()
            train_loss1 += t_loss1.item()
            train_loss2 += t_loss2.item()   # visual loss1 and loss2 in train stage
            train_acc += get_acc(output_[0], targets)


        valid_loss = 0
        valid_acc = 0
        if val_loader is not None and cfg.DATASETS.NAME == "CIFAR100":
            model = model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    output_ = model(images)
                    # v_loss = metric_fc(feature, targets)
                    v_loss = loss_fn(output_[0], targets)                       # Only amsoftmax loss is considered here
                    valid_loss += v_loss.item()
                    valid_acc += get_acc(output_[0], targets)

            avg_t_loss = train_loss / len(train_loader)
            avg_t1_loss = train_loss1 / len(train_loader)
            avg_t2_loss = train_loss2 / len(train_loader)
            avg_v_loss = valid_loss / len(val_loader)
            avg_train_acc = train_acc / len(train_loader)
            avg_val_acc = valid_acc / len(val_loader)
            lr = lr_schedule.get_lr()[0]
            epoch_str = f"Epoch {epoch}: Train Loss1: {avg_t1_loss}, Train Loss2: {avg_t2_loss}, Train Loss: {avg_t_loss}, " \
                f"Train Acc: {avg_train_acc}, Valid Loss: {avg_v_loss}, Valid Acc: {avg_val_acc}, LR: {lr} "
            logger.info(epoch_str)
        elif cfg.DATASETS.NAME == "FACE_DATA":
            pass


        torch.save(model.state_dict(), f"{output_dir}/model.pth")  # 一周期保存一次






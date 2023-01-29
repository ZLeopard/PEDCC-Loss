# encoding: utf-8

import argparse
import os
import sys
from os import mkdir

import torch.nn.functional as F
import torch

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer
from modeling import AMSoftmax

from utils.logger import setup_logger

def get_loss_fn(cfg, logger):
    if cfg.DATASETS.NAME == "CIFAR100":
        if cfg.METRIC.NAME == "Softmax":
            loss_fn = torch.nn.CrossEntropyLoss()
            logger.info("Using Softmax")
            return loss_fn
        elif cfg.METRIC.NAME == "PEDCC-Loss":
            loss_fn = AMSoftmax(cfg.METRIC.S, cfg.METRIC.M, is_amp=False)
            logger.info("Using PEDCC-Loss")
            return loss_fn
    elif cfg.DATASETS.NAME == "FACE_DATA":
        pass

def train(cfg, logger):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE

    optimizer, lr_schedule = make_optimizer(cfg, model)
    metric_fc = None
    loss_fn = get_loss_fn(cfg, logger)
    logger.info("----------------------------------------------")
    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    loss_fn2 = torch.nn.MSELoss(reduction='sum')


    do_train(
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
    )


def main():
    parser = argparse.ArgumentParser(description="PEDCC Loss For Classification")
    parser.add_argument(
        "--config_file", default="configs/train_cifar100.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("Training", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, logger)


if __name__ == '__main__':
    main()

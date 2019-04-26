from yacs.config import CfgNode as CN

# Cofig definition

_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 100

# INPUT SETTING
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = 32
_C.INPUT.SIZE_PADDING = 4

_C.INPUT.SIZE_TEST = 32

_C.INPUT.PROB = 0.5       # random flip
_C.INPUT.PIXEL_MEAN = [0.507, 0.487, 0.441]
_C.INPUT.PIXEL_STD = [0.267, 0.256, 0.276]

# DataSet
_C.DATASETS = CN()
_C.DATASETS.NAME = "CIFAR10"  # CIFAR10  FACE_DATA
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()

# Dataloader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0   # set 0 in windows

# Architecture
_C.ARCHI = CN()
_C.ARCHI.NAME = "VGG"   # VGG, RESNET

# Metric_fc
_C.METRIC = CN()
_C.METRIC.NAME = "Softmax"    # "Softmax" "PEDCC-Loss"
_C.METRIC.S = 15.0
_C.METRIC.M = 0.5
_C.METRIC.N = 1             # To add the nonlinear to MSE loss
_C.METRIC.IS_PEDCC = True
_C.METRIC.IS_MSE = True


# Solver
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.LR_SCHDULER = "MultiStepLR"
_C.SOLVER.MILE_STONES = [25, 50, 80, 100]

_C.SOLVER.MAX_EPOCHS = 120

_C.SOLVER.BASE_LR = 0.1      # LR
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.RESUME = 0         # 当训练停止后，重启设定重启迭代次数
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
_C.SOLVER.IMS_PER_BATCH = 256  #256     # 4 1080TI

# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 200  # 200
_C.TEST.WEIGHT = ""

# output dir
_C.OUTPUT_DIR = ""

# LFW Setting
_C.LFW_ROOT = ""
_C.LFW_TEST_LIST = ""


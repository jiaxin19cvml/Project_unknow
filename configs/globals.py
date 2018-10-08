
ROOT_DIR = "/home/jiaxinchen/Project/3DXRetrieval"
DATASET = "SHREC14"
PROJMETHOD = "C4RAND"
MODALITY = "SHAPE"
NUMB_CHANNELS = 4
FLAG_RANDOM_SAMPLING = False

"""
backbone networks
"""
BASE_NETWORK="resnet50"

"""
constants for building training data
"""
FLAG_SUBSTRACT_MEAN=True
FLAG_SHUFFLE=True
BGRIMAGE_MEAN=[123.68, 116.78, 103.94]
INPUT_IMAGESIZE=(224,224)
"""
constants for training and test
"""
BATCH_SIZE=8
INPUT_QUEUE_SIZE = 4*BATCH_SIZE
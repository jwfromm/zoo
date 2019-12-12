from zookeeper import component
from zookeeper.tf import TFDSDataset


@component
class ImageNet(TFDSDataset):
    name = "imagenet2012:5.0.0"
    train_split = "train"
    validation_split = "validation"


@component
class Cifar10(TFDSDataset):
    name = "cifar10"
    train_split = "train"
    validation_split = "test"


@component
class Mnist(TFDSDataset):
    name = "mnist"
    train_split = "train"
    validation_split = "test"

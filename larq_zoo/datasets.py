from zookeeper import component
from zookeeper.tf import TFDSDataset


@component
class ImageNet(TFDSDataset):
    name = "imagenet2012:5.0.*"
    train_split = "train"
    validation_split = "validation"


@component
class Cifar10(TFDSDataset):
    name = "cifar10:3.0.*"
    train_split = "train"
    validation_split = "test"


@component
class Mnist(TFDSDataset):
    name = "mnist:3.0.*"
    train_split = "train"
    validation_split = "test"


@component
class OxfordFlowers(TFDSDataset):
    name = "oxford_flowers102:2.0.*"
    train_split = "train+validation"
    validation_split = "test"

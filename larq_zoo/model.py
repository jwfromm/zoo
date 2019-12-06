from zookeeper.tf import Model


class LarqZooModel(Model):
    include_top: bool = True
    weights: str

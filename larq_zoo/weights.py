from pathlib import Path

from tensorflow import keras
from zookeeper import component


ROOT_URL = "https://github.com/larq/zoo/releases/download/"


def slash_join(*args):
    return "/".join(arg.strip("/") for arg in args)


@component
class Weights:
    model: str
    version: str
    file: str
    file_hash: str

    def get_path(self, cache_dir: str = None):
        url = slash_join(ROOT_URL, self.model + "-" + self.version, self.file)
        cache_subdir = Path("larq/models") / self.model

        return keras.utils.get_file(
            fname=self.file,
            origin=url,
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            file_hash=self.file_hash,
        )

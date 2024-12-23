import numpy as np
from glob import glob


def _path2id(path):
    return path.split('_')[-3][:3]


def recover_ids(data_dir):
    train_ids = [_path2id(f) for f in glob(str(data_dir / "BONBID2023_Train/3LABEL/*.mha"))]
    train_ids = np.array(sorted(train_ids))
    return train_ids

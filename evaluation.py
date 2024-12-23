from scipy.ndimage import zoom
from sklearn.metrics import f1_score, roc_auc_score, log_loss

import torch
import torch.nn.functional as F
import numpy as np

from data import *


def dice(y_pred, y_true, k=1):
    y_pred = y_pred.astype('float32')
    y_true = y_true.astype('float32')
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + k)


def evaluate(model, validation_ids, data_dir, config, device, debug=False):
    model.eval()
    preds, tars = [], []
    for idx in validation_ids:

        # use the volume as the batch
        x = load_mgh_inputs(idx, data_dir=data_dir, config=config, channels_first=True)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        p = F.sigmoid(model(x)).detach().cpu().numpy()
        y = load_mgh_label(idx, data_dir=data_dir, config=config, channels_first=True)

        # scale back to original size
        s = y.shape[-1] / int(config['image_size'])
        p = zoom(p, (1, 1, s, s))

        preds.append(p)
        tars.append(y)

        if debug: break

    # find best_thresh
    best_score, best_thresh = -1, 0
    min_ = min([p.min() for p in preds])
    max_ = max([p.max() for p in preds])
    for t in np.linspace(min_, max_, num=int(config['num_thresh_sweep_steps'])):
        scores = []
        for (p, y) in zip(preds, tars):
            pt = (p > t).astype('float32')
            scores.append(dice(pt, y))

        scores_avg = np.mean(scores)
        if scores_avg > best_score:
            best_score = scores_avg
            best_thresh = t

    return best_score, best_thresh


def evaluate_3d_seg(model, validation_ids, data_dir, config, device, debug=False):
    model.eval()
    preds, tars = [], []
    seq_len, cval = int(config['seq_len']), int(config['background'])

    for idx in validation_ids:

        # use the volume as the batch
        x_vol = load_mgh_inputs(idx, data_dir=data_dir, config=config, channels_first=True)  # (n, 3, s, s)
        x = []
        for idx_ in batched_range(len(x_vol), seq_len):
            x_seq = pad_seq(x_vol[idx_], seq_len, cval)
            x.append(x_seq)
        x = np.array(x, "float32")  # (n/seq_len, seq_len, 3, s, s)
        x = np.transpose(x, (0, 2, 3, 4, 1))

        x = torch.tensor(x, dtype=torch.float32).to(device)  # (n/seq_len, 3, s, s, seq_len)
        p = F.sigmoid(model(x)).detach().cpu().numpy()  # (n/seq_len, 1, s, s, seq_len)
        y = load_mgh_label(idx, data_dir=data_dir, config=config, channels_first=False)  # (n, 1, s, s)

        p = np.concatenate([p[i] for i in range(p.shape[0])], axis=-1)  # (1, s, s, n)
        p = np.transpose(p, (3, 1, 2, 0))
        p = p[:len(y)]

        # scale back to original size
        s = y.shape[-2] / int(config['image_size'])
        p = zoom(p, (1, s, s, 1))

        preds.append(p)
        tars.append(y)

        if debug: break

    # find best_thresh
    best_score, best_thresh = -1, 0
    min_ = min([p.min() for p in preds])
    max_ = max([p.max() for p in preds])
    for t in np.linspace(min_, max_, num=int(config['num_thresh_sweep_steps'])):
        scores = []
        for (p, y) in zip(preds, tars):
            pt = (p > t).astype('float32')
            scores.append(dice(pt, y))

        scores_avg = np.mean(scores)
        if scores_avg > best_score:
            best_score = scores_avg
            best_thresh = t

    return best_score, best_thresh


def evaluate_outcomes(mgh_validation_ids, bch_validation_ids,
                      model, data_dir, config, device, debug=False):
    mgh_labels, bch_labels = create_outcome_maps(data_dir)

    model.eval()
    preds, tars = [], []
    for idx in mgh_validation_ids:
        k = f"MGHNICU_{idx}"
        if k in mgh_labels:
            x = load_mgh_inputs(idx, data_dir=data_dir, config=config, channels_first=True)
            x = torch.tensor(x, dtype=torch.float32).to(device)
            p = F.sigmoid(model(x)).detach().cpu().numpy()
            p = np.max(p)
            y = mgh_labels[k]

            preds.append(p)
            tars.append(y)

        if debug: break

    for idx in bch_validation_ids:
        k = f"BCHNICU_{idx}"
        if k in bch_labels:
            x = load_bch_inputs(idx, data_dir=data_dir, config=config, channels_first=True)
            x = torch.tensor(x, dtype=torch.float32).to(device)
            p = F.sigmoid(model(x)).detach().cpu().numpy()
            p = np.max(p)
            y = bch_labels[k]

            preds.append(p)
            tars.append(y)

        if debug: break

    preds, tars = np.array(preds, "float32").reshape(-1), np.array(tars, "float32").reshape(-1)

    # find best_thresh
    best_score, best_thresh = -1, 0
    min_ = min([p.min() for p in preds])
    max_ = max([p.max() for p in preds])
    for t in np.linspace(min_, max_, num=int(config['num_thresh_sweep_steps'])):
        preds_t = (preds > t).astype('float32')

        score = f1_score(tars, preds_t)
        if score > best_score:
            best_score = score
            best_thresh = t

    nll = -log_loss(tars, preds)
    score = 0.9 * best_score + 0.1*nll
    print(best_score, nll)
    return score, best_thresh

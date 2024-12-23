import numpy as np
from medpy.io import load
from scipy import ndimage

MODE_STATS = {
    'MGH': {
        "1ADC_ss": {"MEAN": 1350.2495, "STD": 428.13467},
        "2Z_ADC": {"MEAN": 0.34669298, "STD": 2.487756}
    },
    'BCH': {
        "1ADC_ss": {"MEAN": 1976.2158, "STD": 735.4577},
        "2Z_ADC": {"MEAN": 0.17169298, "STD": 2.377756}
    }

}


def preprocess(img, config, src=None, input_type=None):
    # mask prevents blending with the background after scaling
    mask = img != 0

    # resize to 'target_size'
    n = img.shape[-1]
    s = int(config['image_size']) / img.shape[1]
    img = ndimage.zoom(img, (s, s, 1), cval=0.0)
    mask = ndimage.zoom(mask.astype('uint8'), (s, s, 1), cval=0.0)

    # ensure background is constant
    img = img * mask.astype(img.dtype)

    assert n == img.shape[-1]

    # set channels last by default
    img = np.transpose(img, [2, 0, 1])

    if input_type is not None:
        # normalise using precomputed stats
        idx = img != 0
        img[idx] = (img[idx] - MODE_STATS[src][input_type]["MEAN"]) / MODE_STATS[src][input_type]["STD"]

        # mask out background values
        cval = int(config['background'])
        idx = np.logical_not(idx)
        img[idx] = cval
        img = np.clip(img, cval, img.max())

    img = np.expand_dims(img, -1)
    return img


# loading and preprocessing
def load_inputs(ss_adc_filename, zadc_filename, src,
                config, return_meta=False, channels_first=False):
    ss_adc, _ = load(ss_adc_filename)
    ss_adc = preprocess(ss_adc, config, src, '1ADC_ss')

    zadc, h = load(zadc_filename)
    zadc = preprocess(zadc, config, src, '2Z_ADC')

    img = np.concatenate([ss_adc, zadc, zadc], axis=-1)
    if channels_first:
        img = np.transpose(img, (0, 3, 1, 2))

    if return_meta:
        return img, h
    return img


def load_mgh_inputs(idx, data_dir, config, return_meta=False, channels_first=False):
    ss_adc = data_dir / f"BONBID2023_Train/1ADC_ss/MGHNICU_{idx}-VISIT_01-ADC_ss.mha"
    zadc = data_dir / f"BONBID2023_Train/2Z_ADC/Zmap_MGHNICU_{idx}-VISIT_01-ADC_smooth2mm_clipped10.mha"
    return load_inputs(ss_adc, zadc, 'MGH', config, return_meta, channels_first)


def load_bch_inputs(idx, data_dir, config, return_meta=False, channels_first=False):
    ss_adc = data_dir / f"BCH_data_Train_release/1ADC_ss/BCHNICU_{idx}-VISIT_01-ADC_ss.mha"
    zadc = data_dir / f"BCH_data_Train_release/2ZMAP/Zmap_BCHNICU_{idx}-VISIT_01-ADC_smooth2mm_clipped10.mha"
    return load_inputs(ss_adc, zadc, 'BCH', config, return_meta, channels_first)


def load_mgh_label(idx, data_dir, config, return_meta=False, channels_first=False):
    label = data_dir / f"BONBID2023_Train/3LABEL/MGHNICU_{idx}-VISIT_01_lesion.mha"

    label, h = load(label)
    label = preprocess(label, config=config)
    label = np.clip(np.rint(label), 0, 1)
    if channels_first:
        label = np.transpose(label, (0, 3, 1, 2))
    if return_meta:
        return label, h
    return label


def load_data_bonbidhie2023(img_ids, data_dir, config):
    # load and preprocess
    x, y = [], []
    cval = int(config['background'])
    for i in img_ids:
        img = load_mgh_inputs(i, data_dir, config)
        if not np.all(img == cval):
            x.append(img)
            y.append(load_mgh_label(i, data_dir, config))

    # concat
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    return x, y


def load_data_bonbidhie2023_3d(img_ids, data_dir, config):
    # load and preprocess
    x, y = [], []
    cval = int(config['background'])
    seq_len = int(config['seq_len'])
    for i in img_ids:
        img = load_mgh_inputs(i, data_dir, config)
        tar = load_mgh_label(i, data_dir, config)
        for idx in batched_range(len(img), seq_len):
            x_seq = pad_seq(img[idx], seq_len, cval)
            y_seq = pad_seq(tar[idx], seq_len, 0)
            x.append(x_seq)
            y.append(y_seq)

    # concat
    x = np.array(x, "float32")
    y = np.array(y, "float32")

    return x, y


def batched_range(n, bs):
    x = list(range(n))
    return [x[i:i + bs] for i in range(0, n, bs)]


def pad_seq(seq, seq_len, cval):
    s = list(seq.shape)
    s[0] = seq_len
    z = np.zeros(s, "float32") + cval
    z[:len(seq)] = seq
    return z


def load_data_bch(img_ids, data_dir, config):
    # load and preprocess
    x = [load_bch_inputs(x, data_dir, config) for x in img_ids]

    # concat
    x = np.concatenate(x, axis=0)

    return x


def create_outcome_maps(data_dir):
    mgh_labels = {k: int(v) for (k, v) in np.load(str(data_dir / "mgh_train.npy"))}
    bch_labels = {k: int(v) for (k, v) in np.load(str(data_dir / "bch_train.npy"))}
    return mgh_labels, bch_labels


def load_data_outcomes(mgh_ids, bch_ids, data_dir, config):
    mgh_labels, bch_labels = create_outcome_maps(data_dir)

    x, y = [], []
    for idx in mgh_ids:
        k = f"MGHNICU_{idx}"
        if k in mgh_labels:
            img = load_mgh_inputs(idx, data_dir, config)
            tar = mgh_labels[k]
            x.append(img)
            y += [tar for _ in img]
        else:
            print("skipping", k)

    for idx in bch_ids:
        k = f"BCHNICU_{idx}"
        if k in bch_labels:
            img = load_bch_inputs(idx, data_dir, config)
            tar = bch_labels[k]
            x.append(img)
            y += [tar for _ in img]
        else:
            print("skipping", k)

    # concat
    x = np.concatenate(x, axis=0)
    y = np.array(y)[:, None]

    return x, y

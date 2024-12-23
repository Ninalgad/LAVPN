from sklearn.model_selection import train_test_split
import numpy as np


def create_mgh_id_split(df, test_size_proportion, random_state):
    train_ids = df['PID'].map(lambda x: x.split('_')[-1]).values
    percentages = df['lesion percentage'].values

    percentage_category = np.digitize(percentages, bins=[0, 1e-6, 0.01, 0.05, .5, 1])
    training_ids = []
    protected_ids = []
    strata = []
    for (i, s) in zip(train_ids, percentage_category):
        if s in [1, 5]:
            protected_ids.append(i)
        else:
            training_ids.append(i)
            strata.append(s)

    n_train_size = int((1 - test_size_proportion) * len(df))
    ids_train, ids_validation = train_test_split(training_ids,
                                                 train_size=max(1, n_train_size - len(protected_ids)),
                                                 random_state=random_state)
    ids_train = ids_train + protected_ids

    return ids_train, ids_validation, protected_ids


def create_bch_id_split(labels, test_size_proportion, random_state):
    ids, tar = labels.T
    ids = [x.split("_")[1] for x in ids]
    tar = [int(y) for y in tar]

    ids_train, ids_validation = train_test_split(ids, stratify=tar,
                                                 test_size=test_size_proportion,
                                                 random_state=random_state)

    return ids_train, ids_validation

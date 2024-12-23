import argparse
from pathlib import Path
from loguru import logger
import torch
import pandas as pd
import configparser

from model import ClassifierModel
from training import train_outcomes
from data import load_data_outcomes
from split import *


def create_ids_splits(debug=False):
    df_train = pd.read_excel(
        args.data_dir / "BONBID2023_Train/BONBID2023_clinicaldata_train.xlsx",
        header=1, index_col='subject_count'
    )
    mgh_ids_train, mgh_ids_validation, _ = create_mgh_id_split(
        df_train, args.test_fraction, args.random_state)
    if debug:
        mgh_ids_train, mgh_ids_validation = mgh_ids_train[:2], mgh_ids_validation[:2]
    logger.info(f"MGH Splits: train: {len(mgh_ids_train)} val: {len(mgh_ids_validation)}")

    bch_labels = np.load(str(args.data_dir / "bch_train.npy"))
    bch_ids_train, bch_ids_validation = create_bch_id_split(bch_labels, args.test_fraction, args.random_state)
    if debug:
        bch_ids_train, bch_ids_validation = bch_ids_train[:2], bch_ids_validation[:2]
    logger.info(f"BCH Splits: train: {len(bch_ids_train)} val: {len(bch_ids_validation)}")

    n_train, n_val = len(mgh_ids_train)+len(bch_ids_train), len(bch_ids_validation)+len(mgh_ids_validation)
    logger.info(f"Total Splits: train: {n_train} val: {n_val}")
    return mgh_ids_train, mgh_ids_validation, bch_ids_train, bch_ids_validation


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mgh_train, mgh_val, bch_train, bch_val = create_ids_splits(args.debug)

    model = ClassifierModel(pretrained=args.pretrained)
    model.to(device)
    logger.info(f"created model")

    pt_model_file = str(args.model_dir / f'model-{args.random_state}.pt')

    if not args.no_transfer:
        chkpt = torch.load(pt_model_file, device)['model_state_dict']
        tag = "encoder.encoder."
        model.encoder.load_state_dict(
            {k.replace(tag, ""): v for (k, v) in chkpt.items() if tag in k}
        )
        del chkpt
        logger.info(f"loaded weights from {pt_model_file}")

    x, y = load_data_outcomes(mgh_train, bch_train, args.data_dir, config)
    logger.info(f"created training data with {len(y)} samples")

    logger.info("Finetuning model")
    res = train_outcomes(x, y, mgh_val, bch_val,
                         model, device, str(args.model_dir / f'outcomes-model-{args.random_state}'),
                         data_dir=args.data_dir, config=config, debug=args.debug)

    logger.success(f"Completed training. Results: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training script")

    parser.add_argument("--model_dir", type=Path, default=".",
                        help="Directory to save the output model weights in h5 format")
    parser.add_argument("--data_dir", type=Path, default=".",
                        help="Path to the raw features")
    parser.add_argument("--config_file", type=Path, default="./config.ini",
                        help="Configuration file containing model & training parameters in a Windows .ini format")
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Represents the proportion of the dataset to include in the test split. "
                             "Between 0.0 and 1.0")
    parser.add_argument("--pretrained", action='store_true',
                        help="Initilize with Imagenet weights")
    parser.add_argument("--no_transfer", action='store_true',
                        help="Dont load pretrained segmentation encoder")
    parser.add_argument("--random_state", type=int, default=2024,
                        help="Controls the randomness")
    parser.add_argument("--debug", action='store_true',
                        help="Run on a small subset of the data for debugging")

    args = parser.parse_args()

    config_parser = configparser.ConfigParser()
    config_parser.read(args.config_file)
    config = config_parser['DEFAULT']

    main(args, config)

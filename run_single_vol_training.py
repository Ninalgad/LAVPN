import argparse
from pathlib import Path
from loguru import logger
import torch
import pandas as pd
import numpy as np
import configparser

from noise import create_noise_transform
from training import *
from split import create_mgh_id_split

import segmentation_models_pytorch_3d as smp
import torch


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nt = create_noise_transform(config['denoising_transform_name'])

    df_train = pd.read_excel(
        args.data_dir / "BONBID2023_Train/BONBID2023_clinicaldata_train.xlsx",
        header=1, index_col='subject_count'
    )
    logger.info(f"Starting training using {len(df_train)} total patients")

    ids_train, ids_validation, protected_ids = create_mgh_id_split(
        df_train, args.test_fraction, args.random_state)
    logger.info(f"Splits: train: {len(ids_train)} (protected: {len(protected_ids)}) val: {len(ids_validation)}")

    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. resnet34
        in_channels=3,  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        strides=((2, 2, 2), (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 1))
    )
    model.to(device)

    if not args.no_denoise:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
        # pretraining
        if args.tsnr:
            logger.info("Pretraining model using the Truncated SNR objective")
            train_denoise_truncated_snr_3d(model, device, ids_train, args.data_dir, nt,
                                           config=config, debug=args.debug)
        else:
            logger.info("Pretraining model")
            train_denoise(model, device, ids_train, args.data_dir, nt,
                          config=config, debug=args.debug)
        for name, param in model.named_parameters():
            param.requires_grad = True

    # finetune
    logger.info("Finetuning model")
    res = train_bonbidhie2023_3d(model, device, str(args.model_dir / f'model-{args.random_state}'),
                                 ids_train, ids_validation,
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
    parser.add_argument("--no_denoise", action='store_true',
                        help="Pretrain the model before finetuning")
    parser.add_argument("--tsnr", action='store_true',
                        help="pretrain using the truncated snr objective")
    parser.add_argument("--random_state", type=int, default=2024,
                        help="Controls the randomness")
    parser.add_argument("--debug", action='store_true',
                        help="Run on a small subset of the data for debugging")

    args = parser.parse_args()

    config_parser = configparser.ConfigParser()
    config_parser.read(args.config_file)
    config = config_parser['DEFAULT']

    main(args, config)

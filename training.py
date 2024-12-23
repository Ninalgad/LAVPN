import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from evaluation import *
from dataloaders import *
from augment import get_transform
from loss import loss_func
from data import load_data_bonbidhie2023, load_data_bonbidhie2023_3d


def train_denoise(model, device, training_ids, data_dir, noising_transform, config, debug=False):
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_denoising_epochs'])
    inp_size = int(config['image_size'])

    if debug:
        batch_size = 2
        training_ids = training_ids[:2]
        num_epochs = 1

    train_loader = DataLoader(
        DenosingDataset(training_ids, data_dir,
                        transform=get_transform(inp_size),
                        noising_transform=noising_transform,
                        config=config),
        batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=float(config['lr_pretrain']))

    def denoise_train_step(batch):
        inp, tar = batch['input'].to(device).float(), batch['target'].to(device).float()
        optimizer.zero_grad()

        outputs = model(inp)

        mask = (inp > int(config['background'])).float()
        loss_ = (tar - outputs) ** 2
        loss_ = torch.sum(loss_ * mask) / (torch.sum(mask) + 1)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, denoise_train_step, debug)


def train_denoise_truncated_snr(model, device, training_ids, data_dir, noising_transform, config, debug=False):
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_denoising_epochs'])
    inp_size = int(config['image_size'])

    if debug:
        batch_size = 2
        training_ids = training_ids[:2]
        num_epochs = 1

    train_loader = DataLoader(
        DenosingDataset(training_ids, data_dir,
                        transform=get_transform(inp_size),
                        noising_transform=noising_transform,
                        config=config),
        batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=float(config['lr_pretrain']))

    def masked_mse(p, y, mask):
        loss_ = (p - y) ** 2
        return torch.sum(loss_ * mask) / (torch.sum(mask) + 1e-5)

    def denoise_train_step(batch):
        z, e = batch['input'].to(device).float(), batch['noise'].to(device).float()
        phi = batch['dist'].to(device).float()
        x = batch['clean'].to(device).float()
        optimizer.zero_grad()

        v = model(z)
        x_hat = torch.cos(phi) * z - torch.sin(phi) * v
        e_hat = torch.sin(phi) * z + torch.cos(phi) * v

        mask = (z > int(config['background'])).float()
        loss_v = masked_mse(e_hat, e, mask)
        loss_x = masked_mse(x_hat, x, mask)
        loss_ = torch.maximum(loss_v, loss_x)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, denoise_train_step, debug)


def train_denoise_truncated_snr_3d(model, device, training_ids, data_dir, noising_transform, config, debug=False):
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_denoising_epochs'])
    inp_size = int(config['image_size'])

    if debug:
        batch_size = 2
        training_ids = training_ids[:2]
        num_epochs = 1

    train_loader = DataLoader(
        VolumeDenosingDataset(training_ids, data_dir,
                              # transform=get_transform(inp_size),
                              noising_transform=noising_transform,
                              config=config),
        batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=float(config['lr_pretrain']))

    def masked_mse(p, y, mask):
        loss_ = (p - y) ** 2
        return torch.sum(loss_ * mask) / (torch.sum(mask) + 1e-5)

    def denoise_train_step(batch):
        z, e = batch['input'].to(device).float(), batch['noise'].to(device).float()
        phi = batch['dist'].to(device).float()
        x = batch['clean'].to(device).float()
        optimizer.zero_grad()

        v = model(z)
        x_hat = torch.cos(phi) * z - torch.sin(phi) * v
        e_hat = torch.sin(phi) * z + torch.cos(phi) * v

        mask = (z > int(config['background'])).float()
        loss_v = masked_mse(e_hat, e, mask)
        loss_x = masked_mse(x_hat, x, mask)
        loss_ = torch.maximum(loss_v, loss_x)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, denoise_train_step, debug)


def train_bonbidhie2023(model, device, output_name, training_ids, validation_ids, data_dir,
                        config, debug=False):
    x, y = load_data_bonbidhie2023(training_ids, data_dir, config)
    return train_finetune(x, y, model, device, output_name, validation_ids, data_dir, config, debug)


def train_finetune(x, y, model, device, output_name, validation_ids, data_dir,
                   config, debug=False):
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_ft_epochs'])
    inp_size = int(config['image_size'])
    val_epoch = int(config['val_epoch'])

    if debug:
        num_epochs = 1
        batch_size = 2
        val_epoch = 0
        validation_ids = validation_ids[:2]

    train_loader = DataLoader(ImageDataset(x, y, transform=get_transform(inp_size)),
                              batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=float(config['lr_finetune']))

    def train_step(batch):
        inp, tar = batch['image'].to(device), batch['label'].to(device).float()
        optimizer.zero_grad()

        outputs = model(inp)
        loss_ = loss_func(outputs, tar)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    best_val, best_thresh = -1, -1
    for epoch in range(num_epochs):

        train_epoch(model, train_loader, train_step, debug)

        gc.collect()

        if epoch >= val_epoch:
            with torch.no_grad():
                val, thresh = evaluate(model, validation_ids, data_dir, config, device, debug)

            if val > best_val:
                best_val = val
                best_thresh = thresh

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val': val,
                    'best_thresh': thresh,
                }, f'{output_name}.pt')
            print(epoch, best_val)

    return best_val, best_thresh


def train_bonbidhie2023_3d(model, device, output_name, training_ids, validation_ids, data_dir,
                           config, debug=False):
    x, y = load_data_bonbidhie2023_3d(training_ids, data_dir, config)
    return train_finetune_3d(x, y, model, device, output_name, validation_ids, data_dir, config, debug)


def train_finetune_3d(x, y, model, device, output_name, validation_ids, data_dir,
                      config, debug=False):
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_ft_epochs'])
    inp_size = int(config['image_size'])
    val_epoch = int(config['val_epoch'])

    if debug:
        num_epochs = 1
        batch_size = 2
        val_epoch = 0
        validation_ids = validation_ids[:2]

    train_loader = DataLoader(VolumeDataset(x, y, ),  # transform=get_transform(inp_size)),
                              batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=float(config['lr_finetune']))

    def train_step(batch):
        inp, tar = batch['image'].to(device), batch['label'].to(device).float()
        optimizer.zero_grad()

        outputs = model(inp)
        loss_ = loss_func(outputs, tar)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    best_val, best_thresh = -1, -1
    for epoch in range(num_epochs):

        train_epoch(model, train_loader, train_step, debug)

        gc.collect()

        if epoch >= val_epoch:
            with torch.no_grad():
                val, thresh = evaluate_3d_seg(model, validation_ids, data_dir, config, device, debug)

            if val > best_val:
                best_val = val
                best_thresh = thresh

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val': val,
                    'best_thresh': thresh,
                }, f'{output_name}.pt')
            print(epoch, best_val)

    return best_val, best_thresh


def train_outcomes(x, y, mgh_val_ids, bch_val_ids,
                   model, device, output_name, data_dir,
                   config, debug=False):
    batch_size = int(config['batch_size'])
    num_epochs = int(config['num_ft_epochs'])
    inp_size = int(config['image_size'])

    if debug:
        num_epochs = 1
        batch_size = 2
        x, y = x[:2], y[:2]

    train_loader = DataLoader(ImageOutcomeDataset(x, y, transform=get_transform(inp_size)),
                              batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=float(config['lr_finetune']))

    def train_step(batch):
        inp, tar = batch['image'].to(device), batch['label'].to(device).float()
        optimizer.zero_grad()

        outputs = model(inp)
        loss_ = F.binary_cross_entropy(F.sigmoid(outputs), tar)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    best_val, best_thresh = -1, -1
    for epoch in range(num_epochs):

        train_epoch(model, train_loader, train_step, debug)

        gc.collect()
        with torch.no_grad():
            val, thresh = evaluate_outcomes(mgh_val_ids, bch_val_ids,
                                            model, data_dir, config, device, debug)

        if val > best_val:
            best_val = val
            best_thresh = thresh

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val': val,
                'best_thresh': thresh,
            }, f'{output_name}.pt')
        print(epoch, best_val)

    return best_val, best_thresh


def train_epoch(model, train_loader, train_step_fn, debug=False):
    model.train()
    train_loss = []

    for batch in tqdm(train_loader, total=len(train_loader)):
        loss = train_step_fn(batch)

        train_loss.append(loss)
        if debug:
            break
    return sum(train_loss) / len(train_loss)

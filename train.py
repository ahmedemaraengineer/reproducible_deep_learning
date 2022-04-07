#!/usr/bin/env python
# coding: utf-8

import torch, torchaudio
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__) # Log anything that you want to see at any part of the code

# The Dataset Class
class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path = Path('data/ESC-50'), 
                 sample_rate: int = 8000, folds = [1]):
        # Load CSV & initialize all torchaudio.transforms
        # Resample -> MelSpectrogram -> AmplitudeToDB
        self.path = path
        
        self.csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        
        self.resample = torchaudio.transforms.Resample(
        orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB()
    
    def __getitem__(self, index):
        # Return (xb, yb) pair (the spectrogram of index, actual class)
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / 'audio' / row['filename'])
        label = row['target']
        xb = self.db(self.melspec(self.resample(wav)))
        return xb, label
    
    def __len__(self):
        # Return Length
        return len(self.csv)



# The Model Class
class AudioNet(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        for key in hparams.keys():
            self.hparams[key] = hparams[key]

        self.conv1 = nn.Conv2d(1, hparams.base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(hparams.base_filters)
        self.conv2 = nn.Conv2d(hparams.base_filters, hparams.base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hparams.base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(hparams.base_filters, hparams.base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hparams.base_filters * 2)
        self.conv4 = nn.Conv2d(hparams.base_filters * 2, hparams.base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(hparams.base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(hparams.base_filters * 4, hparams.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim.lr)
        return optimizer


# The Training Function
# Hydra allows us to decorate any function to take any configurations
# from any configuration file
# Now any configurations that are in default.yaml are accessible through our train function

@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):

    logger.info(cfg)

    # We use folds 1,2,3 for training, 4 for validation, 5 for testing.
    path = Path(get_original_cwd()) / Path(cfg.data.path)
    train_data = ESC50Dataset(path=path, folds=cfg.data.train_folds)
    val_data = ESC50Dataset(path=path, folds=cfg.data.val_folds)
    test_data = ESC50Dataset(path=path, folds=cfg.data.test_folds)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.data.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.data.batch_size)

    pl.seed_everything(cfg.seed)
    audionet = AudioNet(cfg.model)
    trainer = pl.Trainer(**cfg.trainer) ## ** -> converting a dictionary and take every key value inside the dict
    trainer.fit(audionet, train_loader, val_loader)


if __name__ == '__main__':
    train()

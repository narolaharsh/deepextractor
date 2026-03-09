import torch
import pytest

from deepextractor.models import (
    Autoencoder1D,
    Autoencoder2D,
    DnCNN1D,
    ModifiedAutoencoder2D,
    UNET1D,
    UNET2D,
)


def test_unet1d_forward(sample_1d_batch):
    model = UNET1D(in_channels=1, out_channels=1)
    out = model(sample_1d_batch)
    assert out.shape == sample_1d_batch.shape


def test_unet1d_custom_features():
    model = UNET1D(in_channels=1, out_channels=1, features=[32, 64])
    x = torch.randn(1, 1, 1024)
    out = model(x)
    assert out.shape == x.shape


def test_dncnn1d_forward(sample_1d_batch):
    model = DnCNN1D()
    out = model(sample_1d_batch)
    assert out.shape == sample_1d_batch.shape


def test_autoencoder1d_forward(sample_1d_batch):
    model = Autoencoder1D(in_channels=1, out_channels=1)
    out = model(sample_1d_batch)
    assert out.shape == sample_1d_batch.shape


def test_unet2d_forward(sample_2d_batch):
    model = UNET2D(in_channels=2, out_channels=2)
    out = model(sample_2d_batch)
    assert out.shape == sample_2d_batch.shape


def test_autoencoder2d_forward(sample_2d_batch):
    model = Autoencoder2D(in_channels=2, out_channels=2)
    out = model(sample_2d_batch)
    assert out.shape == sample_2d_batch.shape


def test_modified_autoencoder2d_forward():
    model = ModifiedAutoencoder2D(in_channels=2, out_channels=2)
    x = torch.randn(1, 2, 129, 129)
    out = model(x)
    assert out.shape == x.shape

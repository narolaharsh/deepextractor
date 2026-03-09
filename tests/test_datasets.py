import numpy as np
import torch

from deepextractor.data import SpectrogramDataset, TimeSeriesDataset


def test_timeseries_dataset_len(tmp_npy_files):
    inp, tgt = tmp_npy_files
    ds = TimeSeriesDataset(inp, tgt)
    assert len(ds) == 10


def test_timeseries_dataset_getitem_shape(tmp_npy_files):
    inp, tgt = tmp_npy_files
    ds = TimeSeriesDataset(inp, tgt)
    x, y = ds[0]
    assert x.shape == (1, 8192)
    assert y.shape == (1, 8192)
    assert x.dtype == torch.float32


def test_timeseries_dataset_getitem_all(tmp_npy_files):
    inp, tgt = tmp_npy_files
    ds = TimeSeriesDataset(inp, tgt)
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == (1, 8192)


def test_spectrogram_dataset_len(tmp_spec_npy_files):
    inp, tgt = tmp_spec_npy_files
    ds = SpectrogramDataset(inp, tgt)
    assert len(ds) == 4


def test_spectrogram_dataset_getitem_shape(tmp_spec_npy_files):
    inp, tgt = tmp_spec_npy_files
    ds = SpectrogramDataset(inp, tgt)
    x, y = ds[0]
    # 3D input → channel dimension added → (1, 257, 257)
    assert x.shape == (1, 257, 257)
    assert x.dtype == torch.float32

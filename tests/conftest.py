import pickle

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_1d_batch():
    """Minimal (2, 1, 8192) batch for 1D model testing."""
    return torch.randn(2, 1, 8192)


@pytest.fixture
def sample_2d_batch():
    """Minimal (2, 2, 257, 257) batch for 2D spectrogram model testing."""
    return torch.randn(2, 2, 257, 257)


@pytest.fixture
def tmp_npy_files(tmp_path):
    """Create tiny temporary .npy arrays for dataset tests."""
    inputs = np.random.randn(10, 8192).astype(np.float32)
    targets = np.random.randn(10, 8192).astype(np.float32)
    inp_path = tmp_path / "inputs.npy"
    tgt_path = tmp_path / "targets.npy"
    np.save(inp_path, inputs)
    np.save(tgt_path, targets)
    return str(inp_path), str(tgt_path)


@pytest.fixture
def dummy_scaler_path(tmp_path):
    """Write a trivially-fit StandardScaler to a temp .pkl file."""
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, 1))
    path = tmp_path / "scaler_test.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return str(path)


@pytest.fixture
def tmp_spec_npy_files(tmp_path):
    """Create tiny temporary spectrogram .npy arrays (4D: samples, height, width)."""
    inputs = np.random.randn(4, 257, 257).astype(np.float32)
    targets = np.random.randn(4, 257, 257).astype(np.float32)
    inp_path = tmp_path / "spec_inputs.npy"
    tgt_path = tmp_path / "spec_targets.npy"
    np.save(inp_path, inputs)
    np.save(tgt_path, targets)
    return str(inp_path), str(tgt_path)

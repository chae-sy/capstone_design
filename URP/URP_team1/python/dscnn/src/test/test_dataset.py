import pathlib
import random
import torch
import pytest
import dscnn.legacy.dataset as legacy_dataset
import dscnn.legacy.data_loader as legacy_data_loader
import dscnn.dataset as current_dataset
import dscnn.data_loader as current_data_loader
from dscnn.transform import sp_transform


def test_batch_pipeline():
    NUM_NOISE_SAMPLES = 150
    DATA_PATH = pathlib.Path("data")

    torch.manual_seed(42)
    random.seed(42)
    prev_train, prev_test = legacy_dataset.get_train_and_test_set(
        DATA_PATH, NUM_NOISE_SAMPLES
    )

    legacy_train_loader = legacy_data_loader.get_test_loader(
        test_set=prev_test, batch_size=1024, num_workers=0, pin_memory=False
    )

    prev_data, prev_label = next(iter(legacy_train_loader))

    torch.manual_seed(42)
    random.seed(42)
    current_train, current_test = current_dataset.get_train_and_test_set(
        DATA_PATH, NUM_NOISE_SAMPLES
    )

    current_train_loader = current_data_loader.get_test_loader(
        test_set=current_test, batch_size=1024, num_workers=0, pin_memory=False
    )

    current_data, _, current_label, *_ = next(iter(current_train_loader))

    prev_data = sp_transform(prev_data)

    assert len(prev_train) == len(current_train)
    assert prev_train.labels == current_train.labels

    assert torch.equal(prev_label, current_label)
    assert torch.equal(prev_data[0], current_data[0])

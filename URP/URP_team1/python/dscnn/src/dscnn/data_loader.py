import torch
import torch.utils.data
from dscnn.dataset import SpeechCommandsDataset


def get_train_loader(
    train_set: SpeechCommandsDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_test_loader(
    test_set: SpeechCommandsDataset, batch_size: int, num_workers: int, pin_memory: bool
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

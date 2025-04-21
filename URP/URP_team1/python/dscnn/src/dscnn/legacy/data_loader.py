import torch
import torch.utils.data
from dscnn.legacy.dataset import SpeechCommandsDataset


def _pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def _collate_fn(batch):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors.append(waveform)
        targets.append(label)

    tensors = _pad_sequence(tensors)
    targets = torch.tensor(targets)

    return tensors, targets


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
        collate_fn=_collate_fn,
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
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

import torch


def pad_sequence(batch):
    """
    데이터 길이 맞춰주기 위해 Padding을 수행하는 함수
    """
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """
    Data 중 waveform, label(index로 된)를 각각 tensor, target에 추가

    A data tuple has the form:
    waveform, sample_rate, label, speaker_id, utterance_number
    """

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.tensor(targets)

    return tensors, targets

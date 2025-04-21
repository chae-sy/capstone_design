import sys
sys.path.append('/home/jmlee/urp/dscnn/src/dscnn')
# print(sys.path)
# import dscnn
import os
import random
import torchaudio
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from dscnn.label import KEYWORDS, UNKNOWN_LABEL
from dscnn.transform import sp_transform


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        subset: str,
        sample_ratio: float = 0.4,
        noise_label: str = "_background_noise_",
        num_noise_samples: int = 100,
    ):
        self.dataset_path = dataset_path
        self.keywords = KEYWORDS
        self.unknown_label = UNKNOWN_LABEL
        self.noise_label = noise_label
        self.num_noise_samples = num_noise_samples

        self.all_classes = self.unknown_label + self.keywords + [self.noise_label]
        self.subset = subset
        self.sample_ratio = sample_ratio

        self.keywords_to_index = ["unknown"] + self.keywords + [self.noise_label]
        # Load lists for validation and test datasets
        self.validation_list = self._load_file_list(
            self.dataset_path / "validation_list.txt"
        )
        self.testing_list = self._load_file_list(self.dataset_path / "testing_list.txt")

        self.audio_files = []
        self.noise_samples = []
        self.labels = []
        self.background_noises = []
        self.unknown_files = []
        self.sample_rates = []
        self.speaker_ids = []
        self.utterance_numbers = []

        self._load_all_audio_files()
        self._sample_unknown_class_data()
        self._add_sampled_unknown_class_data()
        self._generate_noise_samples()

    def _load_all_audio_files(self):
        for root, dirs, files in os.walk(self.dataset_path):
            label =  os.path.basename(root)
            if label not in self.all_classes:
                continue

            for file in files:
                if file.endswith(".wav"):
                    root = Path(root)  # 문자열을 Path 객체로 변환
                    file_path = root / file  # 경로 결합
                    if label == self.noise_label:
                        self.background_noises.append(file_path)
                    else:
                        if label not in self.keywords:
                            if self._is_in_subset(file_path):
                                self.unknown_files.append((file_path, "unknown"))
                        elif self._is_in_subset(file_path):
                            waveform, sample_rate = torchaudio.load(file_path)
                            waveform = self._pad_waveform(waveform)
                            waveform = sp_transform(waveform)
                            speaker_id, utterance_number = tuple(file.split("_nohash_"))

                            self.speaker_ids.append(speaker_id)
                            self.utterance_numbers.append(utterance_number)
                            self.audio_files.append(waveform)
                            self.sample_rates.append(sample_rate)
                            self.labels.append(label)

    def _sample_unknown_class_data(self):
        if not self.unknown_files:
            return

        first_label = self.keywords[0]
        first_label_count = sum(1 for label in self.labels if label == first_label)

        total_desired_unknowns = 3 * first_label_count

        if total_desired_unknowns < len(self.unknown_files):
            self.unknown_files = random.sample(
                self.unknown_files, total_desired_unknowns
            )

    def _add_sampled_unknown_class_data(self):
        for file_path, label in self.unknown_files:
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = self._pad_waveform(waveform)
            waveform = sp_transform(waveform)
            speaker_id, utterance_number = tuple(file_path.name.split("_nohash_"))

            self.speaker_ids.append(speaker_id)
            self.utterance_numbers.append(utterance_number)
            self.audio_files.append(waveform)
            self.sample_rates.append(sample_rate)
            self.labels.append(label)

    def _generate_noise_samples(self):
        for _ in range(self.num_noise_samples):
            noise_path = random.choice(self.background_noises)
            waveform, sample_rate = torchaudio.load(noise_path)
            max_offset = waveform.size(1) - sample_rate
            offset = random.randint(0, max_offset)
            noise_slice = waveform[:, offset : offset + sample_rate]
            self.noise_samples.append(sp_transform(self._pad_waveform(noise_slice)))

    def _load_file_list(self, file_path: Path) -> set[str]:
        with open(file_path, "r") as f:
            file_list = f.read().splitlines()
        return set(file_list)

    def _is_in_subset(self, file_path):
        relative_path = os.path.relpath(file_path, self.dataset_path)
        if self.subset == "train":
            return (
                relative_path not in self.validation_list
                and relative_path not in self.testing_list
            )
        elif self.subset == "validation":
            return relative_path in self.validation_list
        elif self.subset == "test":
            return relative_path in self.testing_list
        else:
            raise ValueError("Subset must be one of ['train', 'validation', 'test']")

    def _pad_waveform(self, waveform, target_length=16000):
        current_length = waveform.shape[1]
        if current_length < target_length:
            pad_amount = target_length - current_length
            padding = torch.zeros(
                (waveform.shape[0], pad_amount)
            )  # (channels, pad_amount)
            waveform = torch.cat((waveform, padding), dim=1)
        return waveform

    def __len__(self):
        return len(self.audio_files) + self.num_noise_samples

    def __getitem__(self, idx):
        # classes=self.all_classes ## to be fixed
        if idx < len(self.audio_files):
            waveform = self.audio_files[idx]
            label = self.labels[idx]
            label_index = self.keywords_to_index.index(label)  # label to index
            speaker_id = self.speaker_ids[idx]
            utterance_number = self.utterance_numbers[idx]
            sample_rate = self.sample_rates[idx]
        else:
            noise_idx = idx - len(self.audio_files)
            waveform = self.noise_samples[noise_idx - 1]
            sample_rate = waveform.shape[1]
            label = self.noise_label
            label_index = self.keywords_to_index.index(label)  # label to index
            speaker_id = None
            utterance_number = None

        return waveform, sample_rate, label_index


def get_train_and_test_set(
    data_path: Path, num_noise_samples: int
) -> tuple[SpeechCommandsDataset, SpeechCommandsDataset]:
    """
    Get the train and test set for the Speech Commands dataset.

    Returns:
    The train and test set.
    """
    _ = SPEECHCOMMANDS(root=data_path, download=True)

    speech_commands_dataset_path = (
        data_path / "SpeechCommands" / "speech_commands_v0.02"
    )
    train_set = SpeechCommandsDataset(
        speech_commands_dataset_path,
        subset="train",
        num_noise_samples=num_noise_samples,
    )
    test_set = SpeechCommandsDataset(
        speech_commands_dataset_path, subset="test", num_noise_samples=num_noise_samples
    )

    return train_set, test_set

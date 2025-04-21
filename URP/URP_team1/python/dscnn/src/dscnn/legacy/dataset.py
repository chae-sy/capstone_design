import os
import pathlib
import random
import torchaudio
import torch
from torch.utils.data import Dataset
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from dscnn.label import KEYWORDS, UNKNOWN_LABEL


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        subset,
        sample_ratio=0.4,
        noise_label="_background_noise_",
        num_noise_samples=100,
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
            os.path.join(dataset_path, "validation_list.txt")
        )
        self.testing_list = self._load_file_list(
            os.path.join(dataset_path, "testing_list.txt")
        )

        self.audio_files = []
        self.labels = []
        self.background_noises = []
        unknown_files = []

        # Load all audio files and corresponding labels
        for root, dirs, files in os.walk(dataset_path):
            label = os.path.basename(root)
            if label in self.all_classes:
                for file in files:
                    if file.endswith(".wav"):
                        file_path = os.path.join(root, file)
                        if (
                            label == noise_label
                        ):  # noise append -> (txt에 있으면) unknown에 append + (txt에 있으면) audio_file에 append
                            self.background_noises.append(file_path)
                        else:
                            if (
                                label not in self.keywords
                            ):  # 지정 keyword (10개) 중 없는 경우 -> unknown 할당
                                label = "unknown"
                                if self._is_in_subset(file_path):
                                    unknown_files.append((file_path, label))
                            else:
                                if self._is_in_subset(file_path):
                                    self.audio_files.append(file_path)
                                    self.labels.append(label)

        # unknown class data를 sample_ratio(default=0.2) 비율로 sampling
        if unknown_files:
            # First, find the number of files associated with the first keyword
            first_label = self.keywords[0]
            first_label_count = sum(1 for label in self.labels if label == first_label)

            # Set total_desired_unknowns to three times the first label's count
            total_desired_unknowns = 3 * first_label_count

            # Sample the unknown files if necessary
            if total_desired_unknowns < len(unknown_files):
                unknown_files = random.sample(unknown_files, total_desired_unknowns)

        # Add the sampled unknown files to the dataset
        for file_path, label in unknown_files:
            self.audio_files.append(file_path)
            self.labels.append(label)

        # Generate random slices of background noise
        self.noise_samples = []
        for _ in range(num_noise_samples):
            noise_path = random.choice(self.background_noises)
            waveform, sample_rate = torchaudio.load(noise_path)
            max_offset = waveform.size(1) - sample_rate
            offset = random.randint(0, max_offset)
            noise_slice = waveform[:, offset : offset + sample_rate]
            self.noise_samples.append(noise_slice)

    def _load_file_list(self, file_path):
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
            file_path = self.audio_files[idx]
            label = self.labels[idx]
            label_index = self.keywords_to_index.index(label)  # label to index
            waveform, sample_rate = torchaudio.load(file_path)
            waveform = self._pad_waveform(waveform)  # padding
            filename = os.path.basename(file_path)
            speaker_id, utterance_number = tuple(filename.split("_nohash_"))
            utterance_number = utterance_number.split(".")[0]
        else:
            noise_idx = idx - len(self.audio_files)
            waveform = self.noise_samples[noise_idx - 1]
            sample_rate = waveform.shape[1]
            label = self.noise_label
            label_index = self.keywords_to_index.index(label)  # label to index
            speaker_id = None
            utterance_number = None

        return waveform, sample_rate, label_index, speaker_id, utterance_number


def get_train_and_test_set(
    data_path: pathlib.Path, num_noise_samples: int
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

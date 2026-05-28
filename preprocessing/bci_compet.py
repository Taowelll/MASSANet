from glob import glob

import mne
import numpy as np
import scipy
import torch
from braindecode.datautil.preprocess import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datasets.moabb import MOABBDataset
from mne.filter import resample
from tqdm import tqdm

from preprocessing.augmentation import cutcat
from preprocessing.filters import butter_fir_filter, load_filterbank


class BCICompet2aIV(torch.utils.data.Dataset):
    def __init__(self, args):
        import warnings

        warnings.filterwarnings("ignore")
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.use_augmentation = bool(getattr(args, "use_augmentation", not self.is_test))
        self.downsampling = args.downsampling
        self.args = args
        self.data, self.label = self.get_brain_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if self.use_augmentation:
            data, label = self.augmentation(data, label)

        return {"data": data, "label": label}

    def get_brain_data(self):
        filelist = sorted(glob(f"{self.base_path}/*T*.gdf")) if not self.is_test else sorted(glob(f"{self.base_path}/*E*.gdf"))
        label_filelist = sorted(glob(f"{self.base_path}/*T.mat")) if not self.is_test else sorted(glob(f"{self.base_path}/*E.mat"))

        data = []
        label = []

        for idx, filename in enumerate(tqdm(filelist)):
            if idx != self.target_subject:
                continue

            print(f"LOG >>> Filename: {filename}")
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, _ = mne.events_from_annotations(raw)
            raw.load_data()
            raw.filter(0.0, 40.0, fir_design="firwin")
            raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]

            picks = mne.pick_types(
                raw.info,
                meg=False,
                eeg=True,
                eog=False,
                stim=False,
                exclude="bads",
            )

            if not self.is_test:
                event_id = {"769": 7, "770": 8, "771": 9, "772": 10} if idx != 3 else {"769": 5, "770": 6, "771": 7, "772": 8}
            else:
                event_id = {"783": 7}

            epochs = mne.Epochs(
                raw,
                events,
                event_id,
                0.0,
                3.0,
                proj=True,
                picks=picks,
                baseline=None,
                preload=True,
            )

            if self.downsampling:
                epochs = epochs.resample(self.downsampling)

            epochs_data = epochs.get_data() * 1e6
            trials = [
                exponential_moving_standardize(epoch, init_block_size=int(raw.info["sfreq"] * 4))
                for epoch in epochs_data
            ]
            splited_data = np.stack(trials)[:, np.newaxis, ...]
            label_list = scipy.io.loadmat(label_filelist[idx])["classlabel"].reshape(-1) - 1

            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)

        return data, label

    def augmentation(self, data, label):
        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        ratio = np.random.randint(5, 11)
        return cutcat(
            data,
            label,
            self.data[negative_data_index, ...],
            self.label[negative_data_index],
            self.args.num_classes,
            ratio=ratio,
        )


class BCICompet2bIV(torch.utils.data.Dataset):
    def __init__(self, args):
        import warnings

        warnings.filterwarnings("ignore")
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.use_augmentation = bool(getattr(args, "use_augmentation", not self.is_test))
        self.downsampling = args.downsampling
        self.args = args
        self.data, self.label = self.get_brain_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if self.use_augmentation:
            data, label = self.augmentation(data, label)

        return {"data": data, "label": label}

    def get_brain_data(self):
        filelist = sorted(glob(f"{self.base_path}/*T.gdf")) if not self.is_test else sorted(glob(f"{self.base_path}/*E.gdf"))
        label_filelist = sorted(glob(f"{self.base_path}/*T.mat")) if not self.is_test else sorted(glob(f"{self.base_path}/*E.mat"))

        data = []
        label = []

        for idx, filename in enumerate(tqdm(filelist)):
            if not self.is_test:
                if idx // 3 != self.target_subject:
                    continue
            elif idx // 2 != self.target_subject:
                continue

            print(f"LOG >>> Filename: {filename}")
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)
            raw.load_data()
            raw.filter(0.0, 40.0, fir_design="firwin")
            raw.info["bads"] += ["EOG:ch01", "EOG:ch02", "EOG:ch03"]

            picks = mne.pick_types(
                raw.info,
                meg=False,
                eeg=True,
                eog=False,
                stim=False,
                exclude="bads",
            )

            event_id = {"769": annot["769"], "770": annot["770"]} if not self.is_test else {"783": annot["783"]}
            epochs = mne.Epochs(
                raw,
                events,
                event_id,
                0.0,
                3.0,
                proj=True,
                picks=picks,
                baseline=None,
                preload=True,
            )

            if self.downsampling:
                epochs = epochs.resample(self.downsampling)

            epochs_data = epochs.get_data() * 1e6
            trials = [
                exponential_moving_standardize(epoch, init_block_size=int(raw.info["sfreq"] * 4))
                for epoch in epochs_data
            ]
            splited_data = np.stack(trials)[:, np.newaxis, ...]
            label_list = scipy.io.loadmat(label_filelist[idx])["classlabel"].reshape(-1) - 1

            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)

        return data, label

    def augmentation(self, data, label):
        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        return cutcat(
            data,
            label,
            self.data[negative_data_index, ...],
            self.label[negative_data_index],
            self.args.num_classes,
            ratio=10,
        )


class OpenBMI(torch.utils.data.Dataset):
    def __init__(self, args):
        import warnings

        warnings.filterwarnings("ignore")
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.use_augmentation = bool(getattr(args, "use_augmentation", not self.is_test))
        self.downsampling = args.downsampling
        self.args = args
        self.data, self.label = self.get_brain_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if self.use_augmentation:
            data, label = self.augmentation(data, label)

        return {"data": data, "label": label}

    @staticmethod
    def _resolve_session_keys(sessions):
        possible_train_keys = ["session1", "session_1", "train", "session_train"]
        possible_test_keys = ["session2", "session_2", "test", "session_test"]

        train_key = next((key for key in possible_train_keys if key in sessions), None)
        test_key = next((key for key in possible_test_keys if key in sessions), None)

        available_keys = list(sessions.keys())
        if train_key is None and available_keys:
            train_key = available_keys[0]
        if test_key is None and len(available_keys) > 1:
            test_key = available_keys[1]
        if test_key is None:
            test_key = train_key
        return train_key, test_key

    def get_brain_data(self):
        dataset = MOABBDataset(dataset_name="Lee2019_MI", subject_ids=[self.target_subject + 1])

        preprocessors = [
            Preprocessor(fn="pick_types", eeg=True, meg=False, stim=False, apply_on_array=True),
            Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),
            Preprocessor(fn="filter", l_freq=0.0, h_freq=40.0, apply_on_array=True),
            Preprocessor(
                fn=exponential_moving_standardize,
                factor_new=1e-3,
                init_block_size=1000,
                apply_on_array=True,
            ),
        ]
        preprocess(dataset, preprocessors)

        sfreq = dataset.datasets[0].raw.info["sfreq"]
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            preload=True,
        )

        sessions = windows_dataset.split("session")
        train_key, test_key = self._resolve_session_keys(sessions)
        session_key = test_key if self.is_test else train_key

        x_list = []
        y_list = []
        for trial in sessions[session_key]:
            x_list.append(trial[0])
            y_list.append(trial[1])

        x_list = np.array(x_list)
        y_list = np.array(y_list)
        x_list = x_list[..., : int(3.0 * sfreq)]

        if self.downsampling and self.downsampling != sfreq:
            down = int(round(sfreq / self.downsampling))
            x_list = resample(np.asarray(x_list, dtype=np.float64), down=down)

        data = x_list[:, np.newaxis, 20:40, :]
        label = np.asarray(y_list)
        return data, label

    def augmentation(self, data, label):
        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        return cutcat(
            data,
            label,
            self.data[negative_data_index, ...],
            self.label[negative_data_index],
            self.args.num_classes,
            ratio=10,
        )


def _apply_filter_bank(dataset, args):
    data = dataset.data
    bands = args["bank"]
    sampling_rate = args.downsampling if args.downsampling else args.sampling_rate
    data_filterbank = np.zeros(
        (data.shape[0], data.shape[1], len(bands) * data.shape[2], data.shape[3]),
        dtype=data.dtype,
    )

    for band_idx, freq_band in enumerate(bands):
        filter_coef = load_filterbank(np.array(freq_band), sampling_rate, order=4, max_freq=40, ftype="butter")
        filtered = np.zeros_like(data)
        for trial_idx, trial in enumerate(data):
            filtered_trial = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            filtered[trial_idx, :, :, :] = filtered_trial.reshape(1, data.shape[2], data.shape[3])

        start = band_idx * data.shape[2]
        end = (band_idx + 1) * data.shape[2]
        data_filterbank[:, :, start:end, :] = filtered

    dataset.data = data_filterbank
    return dataset


def get_dataset(config_name, args):
    config_name = config_name.lower()

    if "bcicompet2a_config" in config_name:
        dataset = BCICompet2aIV(args)
    elif "bcicompet2b_config" in config_name:
        dataset = BCICompet2bIV(args)
    elif "kumi_config" in config_name or "openbmi_config" in config_name:
        dataset = OpenBMI(args)
    else:
        raise ValueError("Unsupported dataset config.")

    if args["filter_bank"]:
        dataset = _apply_filter_bank(dataset, args)

    return dataset

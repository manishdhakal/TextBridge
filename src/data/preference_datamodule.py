from typing import Any, Dict, Optional, Tuple
import os

import pandas as pd
import numpy as np
import av

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from lightning import LightningDataModule
from transformers import AutoProcessor


class MUStARDPreferenceDataset(Dataset):
    def __init__(
        self,
        preference_data_path: str,
        video_dir: str,
        num_frames: int,
        num_input_tokens: int = 1170,
        num_output_tokens: int = 500,
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames

        self.description = pd.read_csv(preference_data_path)
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf"
        )

        self.query = "Describe the video in details"
        self.prompt = f"USER: <video>\n{self.query} ASSISTANT:"

        # Given the constant video input and prompt sizes, we know the number of input tokens
        # TODO: Change this according to your input size
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens

    def __len__(self):
        return len(self.description)
    
    def _read_video_pyav(self, container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    def __getitem__(self, idx):
        row = self.description.iloc[idx]
        video_id = row["video_id"]
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
        clip = self._read_video_pyav(container, indices)
        inputs_video = self.processor(
            text=self.prompt,
            videos=clip,
            return_tensors="pt",
            max_length=self.num_input_tokens,
            truncation=True,
        )
        
        eos_token = self.processor.tokenizer.eos_token
        preferred_desc, dispreferred_desc = (
            row["preferred_description"] + eos_token,
            row["dispreferred_description"] + eos_token,
        )

        gt_desc = self.processor.tokenizer(
            [preferred_desc, dispreferred_desc],
            return_tensors="pt",
            max_length=self.num_output_tokens,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            padding_side="right",
        )
        
        return {
            "input_ids": inputs_video["input_ids"].squeeze(0),
            "attention_mask": inputs_video["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs_video["pixel_values_videos"].squeeze(0),
            "preferred_ids": gt_desc["input_ids"][0],
            "dispreferred_ids": gt_desc["input_ids"][1],
        }


class PreferenceDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MUStARDPreferenceDataset is a dataset for video preference learning. 
    It contains a collection of videos and their corresponding preferred and dispreferred descriptions.
    The dataset is used to train a model to learn the preference between two descriptions of the same video.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        preference_data_path: str,
        video_dir: str,
        num_frames: int,
        num_input_tokens: int = 1170,
        num_output_tokens: int = 500,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `PreferenceDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = MUStARDPreferenceDataset(
            preference_data_path=self.hparams.preference_data_path,
            video_dir=self.hparams.video_dir,
            num_frames=self.hparams.num_frames,
            num_input_tokens=self.hparams.num_input_tokens,
            num_output_tokens=self.hparams.num_output_tokens,
        )
        
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None and self.trainer.strategy == "ddp":
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        #     trainset = MNIST(
        #         self.hparams.data_dir, train=True, transform=self.transforms
        #     )
        #     testset = MNIST(
        #         self.hparams.data_dir, train=False, transform=self.transforms
        #     )
        #     dataset = ConcatDataset(datasets=[trainset, testset])
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset=dataset,
        #         lengths=self.hparams.train_val_test_split,
        #         generator=torch.Generator().manual_seed(42),
        #     )
        
        pass
        

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # return DataLoader(
        #     dataset=self.data_val,
        #     batch_size=self.batch_size_per_device,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=False,
        # )
        return None
        

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        
        # return DataLoader(
        #     dataset=self.data_test,
        #     batch_size=self.batch_size_per_device,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=False,
        # )
        return None

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    # _ = MNISTDataModule()
    pass

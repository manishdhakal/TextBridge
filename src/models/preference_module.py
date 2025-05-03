from typing import Any, Dict, Tuple
import copy

import torch
from torch import nn
from torch.nn import functional as F

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model

# from bitsandbytes import  Bit
from .loss import DPOLoss

lora_cofig = LoraConfig(
    r=1,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.0,
    bias="none",
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)


class LogProbability(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Calculate the log probability of the labels given the logits.
        Args:
            logits (`torch.Tensor`): The logits from the model.
            labels (`torch.LongTensor`): The labels for which to calculate the log probability.
        Returns:
            `torch.Tensor`: The log probabilities of the labels.
        """
        # Dimensions check, labels must one dim less than logits
        if labels.dim() != logits.dim() - 1:
            raise ValueError(
                f"Expected labels of dimension {logits.dim() - 1} which is one dim less than logits, but got {labels.dim()}"
            )

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=self.dim)

        # Gather the probabilities for the true labels
        gathered_probs = probs.gather(self.dim, labels.unsqueeze(self.dim)).squeeze(
            self.dim
        )

        # Calculate log probabilities
        log_probs = torch.log(gathered_probs)

        return log_probs


class PreferenceModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code xhere.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_output_tokens: int,
        compile: bool,
    ) -> None:
        """Initialize a `PreferenceModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.ref_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            # device_map="",
        )

        self.policy_model = copy.deepcopy(self.ref_model)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # policy_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        #     "llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16
        # )
        self.policy_model = get_peft_model(
            self.policy_model,
            lora_cofig,
        )
        self.policy_model.print_trainable_parameters()

        # calculate the log probability of all generated tokens
        self.compute_logps = LogProbability(dim=-1)

        self.dpo_loss = DPOLoss(beta=0.1)

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def _update_positional_and_cache_ids(self, inputs_video: Dict, first_input: bool):
        """
        Update the positional ids of the video.
        Args:
            inputs_video (`Dict`): Dictionary containing the input tensors.
            first_input (`bool`): Whether this is the first input or not in auto-regressive generation.
        Returns:
            inputs_video (`Dict`): Updated dictionary with new positional ids.
        """
        device = inputs_video["input_ids"].device
        inputs_video = dict(inputs_video)
        if first_input:
            batch_size, num_tokens = inputs_video["input_ids"].shape[:2]
            ids = torch.arange(num_tokens, device=device)
            inputs_video["cache_position"] = ids
            inputs_video["postional_ids"] = ids.expand(batch_size, num_tokens)
        else:
            batch_size = inputs_video["input_ids"].shape[0]
            ids = torch.max(inputs_video["postional_ids"]) + 1
            inputs_video["cache_position"] = ids.expand(batch_size)
            inputs_video["postional_ids"] = ids.expand(batch_size, 1)

        return inputs_video

    def _prepare_inputs_for_generation(
        self, inputs_video: Dict, predicted_outputs=None
    ):
        """
        Prepare the inputs for generation.
        Args:
            inputs_video (`Dict`): Dictionary containing the input tensors.
            predicted_outputs (`Dict | None`): The predicted outputs from the model with `inputs_video`.
                Contains `logits` and `past_key_values`. None if this is the first input.
        Returns:
            inputs_video (`Dict`): Updated dictionary with new input tensors.
        """
        device = inputs_video["input_ids"].device
        inputs_video = dict(inputs_video)

        if predicted_outputs is None:
            inputs_video = self._update_positional_and_cache_ids(
                inputs_video, first_input=True
            )
            inputs_video["past_key_values"] = None
            inputs_video["logits_to_keep"] = 1
            inputs_video["use_cache"] = True
        else:
            inputs_video["input_ids"] = predicted_outputs["logits"].argmax(dim=-1)
            inputs_video["attention_mask"] = torch.cat(
                [
                    torch.ones(
                        (inputs_video["attention_mask"].shape[0], 1), device=device
                    ),
                    inputs_video["attention_mask"],
                ],
                dim=1,
            )
            inputs_video = self._update_positional_and_cache_ids(
                inputs_video, first_input=False
            )
            inputs_video["past_key_values"] = predicted_outputs["past_key_values"]
            inputs_video["logits_to_keep"] = 1
            inputs_video["use_cache"] = True
            inputs_video["pixel_values_videos"] = None
        return inputs_video

    def calculate_dpo_loss(self, **inputs_video) -> torch.Tensor:
        preferred_ids, dispreferred_ids = (
            inputs_video["preferred_ids"],
            inputs_video["dispreferred_ids"],
        )
        ref_inputs = inputs_video.copy()

        with torch.no_grad():
            predicted_outputs = None
            logits = []
            for i in range(self.hparams.num_output_tokens):
                ref_inputs = self._prepare_inputs_for_generation(
                    ref_inputs, predicted_outputs=predicted_outputs
                )
                predicted_outputs = self.ref_model(**ref_inputs)
                logits.append(predicted_outputs["logits"])
            logits = torch.cat(logits, dim=1)

        reference_prefered_logps = self.compute_logps(logits, preferred_ids)
        reference_disprefered_logps = self.compute_logps(logits, dispreferred_ids)

        policy_inputs = inputs_video
        predicted_outputs = None
        logits = []
        for _ in range(self.hparams.num_output_tokens):
            policy_inputs = self._prepare_inputs_for_generation(
                policy_inputs, predicted_outputs=predicted_outputs
            )
            predicted_outputs = self.policy_model(**policy_inputs)
            logits.append(predicted_outputs["logits"])
        logits = torch.cat(logits, dim=1)
        policy_prefered_logps = self.compute_logps(logits, preferred_ids)
        policy_disprefered_logps = self.compute_logps(logits, dispreferred_ids)

        # calculate the loss
        loss = self.dpo_loss(
            policy_prefered_logps,
            policy_disprefered_logps,
            reference_prefered_logps,
            reference_disprefered_logps,
        )[0]
        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss = self.calculate_dpo_loss(**batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log(
            "train/dpo_loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.ref_model = torch.compile(self.ref_model)
            self.policy_model = torch.compile(self.policy_model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.policy_model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # _ = MNISTLitModule(None, None, None, None)
    pass

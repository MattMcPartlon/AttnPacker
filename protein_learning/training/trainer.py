"""Train, Test, and Validate a Model"""
# import os
import time
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import Adam, AdamW

from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.data.data_types.model_loss import ModelLoss
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.data.datasets.dataset import ProteinDatasetABC
from protein_learning.common.global_config import GlobalConfig
from protein_learning.common.global_constants import get_logger, get_current_time
from protein_learning.common.helpers import exists, default, time_fn
from protein_learning.models.model_abc.protein_model import ProteinModel
from protein_learning.training.train_utils import (
    check_nan_in_grad,
    EvalTy,
    get_grad_norm,
    safe_round,
    num_trainable_params,
    Queue,
)
from copy import deepcopy

logger = get_logger(__name__)


class Trainer:
    """Model Trainer"""

    def __init__(self,
                 config: GlobalConfig,
                 model: ProteinModel,
                 train_data: ProteinDatasetABC,
                 valid_data: Optional[ProteinDatasetABC] = None,
                 test_data: Optional[ProteinDatasetABC] = None,
                 rolling_window_size: Optional[int] = None,
                 force_load_path: str = None,  # force loading of a model given path
                 post_init_fn: Optional = None,
                 allow_missing_modules: bool = False,
                 load_optim: bool = True,
                 ):
        self.config = config
        self.allow_missing_modules = allow_missing_modules
        data_keys = [EvalTy.TRAINING.value, EvalTy.VALIDATION.value, EvalTy.TESTING.value]
        self.datasets = {k: v for k, v in zip(data_keys, [train_data, valid_data, test_data])}
        self.sample_idx, self.checkpoint, self.batch_idx = 0, 0, 0
        self.epoch, self.n_processed, self.n_valid_batch_samples = 0, 0, 0
        self.load_optim = load_optim

        # set up model and optimizer
        self.model = model.to(config.device)
        if len(config.gpu_indices) > 1:
            self.model.set_model_parallel(config.gpu_indices)
        self.optim = self._get_optim(model=self.model, config=config)
        self.epoch_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lambda x: config.decrease_lr_by ** max(1, x), last_epoch=-1)

        # optimize over per-sample gradients or average over batch
        self.use_amp = not config.clip_grad_per_sample
        self.decrease_lr_every = config.decrease_lr_every if config.decrease_lr_every > 0 \
            else config.epochs + 1
        self.clipped_grad = None
        rolling_window_size = default(rolling_window_size, config.batch_size * 100)
        self.rolling_window_loss = Queue(rolling_window_size)
        if exists(force_load_path):
            self._init_model(force_load_path)
        self.post_init_fn = post_init_fn

    def pre_sample(self, model_in: ModelInput) -> ModelInput:  # noqa
        """perform any pre-sample processing steps"""
        return model_in

    def post_sample(  # noqa
            self,
            model_in: ModelInput,
            model_out: ModelOutput,
            model_loss: ModelLoss
    ) -> Tuple[ModelOutput, ModelLoss]:
        """perform any post-sample processing steps"""
        if model_loss is not None:
            self.rolling_window_loss.append(model_loss.float_value)
        return model_out, model_loss

    def pre_batch(self):
        """Perform any pre-batch processing steps"""
        pass

    def clear_grad_dict(self):
        """Clear per-sample gradient dict"""
        if self.config.clip_grad_per_sample:
            self.clipped_grad = {name: 0 for name, param in self.model.named_parameters()}

    def post_batch(self):
        """Perform any post-batch processing steps"""
        self.clear_grad_dict()

    def train(self):
        """Train Model"""
        logger.info("Beginning Model Training")
        self._init_model()
        config, start, cum_time, _n_processed = self.config, None, 0, 0

        for epoch in range(config.epochs):
            logger.info(f"Beginning training epoch : {epoch}")
            data = self.datasets[EvalTy.TRAINING.value]
            if config.shuffle:
                data.shuffle()  # to be safe :)
            train_data_loader = data.get_dataloader(num_workers=config.data_workers,
                                                    prefetch_factor=2,
                                                    batch_size=min(config.batch_size, 8),
                                                    shuffle=config.shuffle,
                                                    )
            lag = time.time()
            for _, samples in enumerate(train_data_loader):
                start = default(start, time.time())
                logger.info(f'[batch {self.batch_idx}], loaded {len(samples)} samples')
                print(f"[INFO] Loaded batch of {len(samples)} "
                      f"training examples (lag = {np.round(time.time() - lag, 2)}s)")
                for sample in samples:
                    # Setup for batch
                    self.batch_idx = self.n_processed // config.batch_size
                    if self.n_processed % config.batch_size == 0:
                        self.pre_batch()

                    with autocast(enabled=self.use_amp):
                        sample = self.pre_sample(sample)
                        model_out, model_loss = self.process_sample(
                            sample,
                            eval_ty=EvalTy.TRAINING,
                        )
                    if model_out is None:
                        continue

                    model_out, model_loss = self.post_sample(sample, model_out, model_loss)
                    # accumulate gradients for sample
                    self.accum_grad(
                        model_loss=model_loss,
                        model=self.model,
                    )
                    self.n_processed += 1

                    # display loss
                    if self.n_processed % (config.batch_size // 2) == 0:
                        self._display_loss(model_loss, eval_ty=EvalTy.TRAINING)

                    model_loss, model_out, _n_processed = None, None, _n_processed + 1

                    # Update model
                    if self.n_processed % config.batch_size == 0:
                        self.update_model(self.model)
                        self.post_batch()
                        cum_time += time.time() - start
                        avg_time = np.round(cum_time / _n_processed, 3)
                        print("##### average time per sample :", avg_time, "#####")
                        print(f"##### CURRENT TIME : {get_current_time()} #####")
                        start = time.time()

                    # save the model
                    if self.n_processed % int(config.save_every * config.batch_size) == 0:
                        self.save_state()

                    # checkpoint model
                    if self.n_processed % int(config.checkpoint_every * config.batch_size) == 0:
                        self.checkpoint_model()

                    # run on validation targets
                    if self.n_processed % int(config.validate_every * config.batch_size) == 0:
                        if self.datasets[EvalTy.VALIDATION.value] is not None:
                            self._process_dataset(
                                eval_ty=EvalTy.VALIDATION,
                                max_samples=config.max_val_samples,
                            )
                            start = time.time()

                    # run on test targets
                    if self.n_processed % int(config.test_every * config.batch_size) == 0:
                        if self.datasets[EvalTy.TESTING.value] is not None:
                            self._process_dataset(
                                eval_ty=EvalTy.TESTING,
                                max_samples=config.max_test_samples
                            )
                            start = time.time()
                lag = time.time()

            self.epoch += 1
            if (self.epoch % self.decrease_lr_every) == 0:
                self.epoch_scheduler.step()
            self.sample_idx = 0

    def update_model(
            self,
            model: nn.Module,
            optim: Optional[torch.optim.Optimizer] = None,
            grad_dict: Optional[Dict] = None,
            scaler: Optional = None,
            model_name: str = "",
    ) -> None:
        """Update model (optim.step)"""
        optim = default(optim, self.optim)
        grad_dict = default(grad_dict, self.clipped_grad)

        config, updated = self.config, 0
        if config.clip_grad_per_sample:
            denom = config.batch_size if config.average_batch else 1
            for name, param in model.named_parameters():
                param_grad = grad_dict[name] if name in grad_dict else None
                if param_grad is not None and torch.is_tensor(param_grad):
                    param.grad = param_grad / denom

        # global clip
        if not config.clip_grad_per_sample:
            scaler.unscale_(optim)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)
        gradient_norm = get_grad_norm(model)
        has_nan = check_nan_in_grad(model)
        if not has_nan and not torch.isnan(gradient_norm):
            scaler.step(optim) if exists(scaler) else optim.step()
            scaler.update() if exists(scaler) else None
        else:
            logger.warn(f'Caught NaN in optim.step() {model_name}! skipping...')
        optim.zero_grad()
        print(f"[INFO] global gradient norms {model_name} "
              f"{safe_round(gradient_norm)} at batch : {self.batch_idx}")

    def process_sample(self,
                       sample: ModelInput,
                       eval_ty: EvalTy,
                       ) -> Tuple[Optional[ModelOutput], Optional[ModelLoss]]:
        """Run model on a single sample

        (1)  evaluate the model
        (2)  get the sample loss
        """
        forward_time, model_out = time_fn(self.safe_eval, sample, eval_ty=eval_ty)

        if not exists(model_out):
            return None, None

        bw_time, model_loss = time_fn(
            self.get_model_loss,
            model_out=model_out,
            eval_ty=eval_ty,
        )

        if self.n_processed % (self.config.batch_size // 2) == 0:
            logger.info(f"time for forward pass : {np.round(forward_time, 3)}")
            logger.info(f"time for backward pass : {np.round(bw_time, 3)}")

        return model_out, model_loss

    def safe_eval(
            self,
            sample: ModelInput,
            eval_ty: EvalTy,
            raise_exceptions: Optional[bool] = None,
            use_amp: Optional[bool] = None
    ) -> ModelOutput:
        """Evaluate model and (optionally) catch exceptions"""
        with autocast(enabled=default(use_amp,self.use_amp)):
            with torch.set_grad_enabled(eval_ty == eval_ty.TRAINING):
                try:
                    return self.model(sample.to(self.config.device).crop(self.config.max_len))
                except Exception as e:
                    logger.error(f'Caught exception {e} evaluating model,'
                                f' eval_ty :{eval_ty.value}, sample_idx : {self.n_processed}')
                    if default(raise_exceptions, self.config.raise_exceptions):
                        raise e

    def get_model_loss(
            self,
            model_out: ModelOutput,
            eval_ty: EvalTy,
    ) -> Optional[ModelLoss]:
        """Compute model loss"""
        logger.info("Getting Model Loss")
        compute_zero_wt_loss = (self.n_processed + 1) % (self.config.batch_size // 2) == 0
        with torch.set_grad_enabled(eval_ty == eval_ty.TRAINING):
            try:
                return self.model.compute_loss(
                    output=model_out,
                    compute_zero_wt_loss=compute_zero_wt_loss,
                    batch_idx=self.batch_idx
                )
            except Exception as e:
                logger.error(f"Error {e} calculating model loss, eval_ty : {eval_ty.value}")
                if self.config.raise_exceptions:
                    raise e
                return None

    """Optimization Specific Functions"""

    def accum_grad(
            self,
            model: nn.Module,
            model_loss: Optional[ModelLoss],
            grad_dict: Optional[Dict] = None,
            scaler: Optional = None,
            retain_graph: bool = False,
    ) -> Optional[Dict]:
        """Accumulate gradient"""
        logger.info("Accumulating Model Gradients")
        if model_loss is None or model is None:
            return

        loss, config = model_loss.get_loss(), self.config
        grad_dict = default(grad_dict, self.clipped_grad)
        # check for nan in gradient and skip if found
        if check_nan_in_grad(model) or torch.any(torch.isnan(loss)):
            logger.warn("found NaN in gradients. skipping update")
            model.zero_grad()
            return

        # self.scaler used for mixed precision training
        denom = 1 if config.clip_grad_per_sample else config.batch_size
        if exists(scaler):
            scaler.scale(loss / denom).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)  # noqa

        if config.clip_grad_per_sample:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.example_clip)
            grad_params = filter(lambda x: x[1] is not None and x[1].grad is not None,
                                 model.named_parameters())

            for name, param in grad_params:
                grad_dict[name] += param.grad
            model.zero_grad()
        return grad_dict

    """I/O Methods"""

    def checkpoint_model(self, **kwargs):
        """Checkpoint (Hard Copy) model state"""
        logger.info("Checkpointing Model State")
        """hard-checkpoint of model state - in case things go awry"""
        self.save_state(path=self.config.checkpoint_path(self.checkpoint), **kwargs)
        self.checkpoint += 1

    def save_state(self, path: Optional[str] = None, **kwargs) -> None:
        """Save model and training state"""
        print("[INFO] Saving Model State")
        to_save = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch_scheduler': self.epoch_scheduler.state_dict(),
            'epoch': self.epoch,
            'processed': self.n_processed,
            'sample_idx': self.sample_idx,
            'checkpoint': self.checkpoint,
            'batch_idx': self.batch_idx,
        }
        to_save.update(kwargs)
        torch.save(
            to_save,
            default(path, self.config.save_path)
        )

    def load_state(self, path: Optional[str] = None):
        """Load from previous state"""
        config, path = self.config, default(path, self.config.model_load_path)
        print(f"[INFO] LOADING MODEL \nCheckpoint = {config.load_from_checkpoint}")
        logger.info(f"LOADING MODEL, Checkpoint = {config.load_from_checkpoint}")
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f'[ERROR] Failed to load model! {e}\n path : {path}')
            raise e
        print(f"[INFO]: Loading from path {path}")
        if self.allow_missing_modules:
            print("[INFO] allowing missing modules loading state dict")
        state_dict = self.perturb_wts(checkpoint['model'], perturb_eps=self.config.perturb_wts)
        self.model.load_state_dict(state_dict, strict=not self.allow_missing_modules)
        if not self.allow_missing_modules and self.load_optim:
            print("[INFO] Loading previous optimizer state...")
            self.optim.load_state_dict(checkpoint['optim'])
        self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
        self.epoch = checkpoint['epoch']
        self.n_processed = checkpoint['processed']
        self.sample_idx = checkpoint['sample_idx']
        self.checkpoint = checkpoint['checkpoint']
        self.batch_idx = checkpoint['batch_idx']
        print(f"[INFO] Starting from batch index {self.batch_idx}")
        return checkpoint

    """Private Helper Methods"""

    def _display_loss(self, loss: ModelLoss, eval_ty: EvalTy):
        """Display model loss"""
        if not (exists(loss)):
            return
        print(f"###### BATCH_IDX {self.batch_idx}, EPOCH {self.epoch},"
              f" TY {eval_ty.value}, PROCESSED {self.n_processed} ######")
        loss.display_loss()
        print(f"[INFO] Rolling Average w={self.rolling_window_loss.max_len}:"
              f"  {np.round(np.mean(self.rolling_window_loss.data), 3)}")

    def _process_dataset(
            self,
            eval_ty: EvalTy = EvalTy.VALIDATION,
            dataset: ProteinDatasetABC = None,
            max_samples: int = -1,
    ):
        """
        Run model on all samples in a dataset - does not compute gradients.
        """
        logger.info(f"Processing {eval_ty.value} dataset")
        dataset = default(dataset, self.datasets[eval_ty.value])
        data = dataset.get_dataloader(
            batch_size=min(len(dataset), 8),
            num_workers=2,
            shuffle=False
        )
        n_processed, start = 0, time.time()
        self.model = self.model.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(data):
                if n_processed > max_samples > 0:
                    break
                for sample in batch_data:
                    model_out, model_loss = self.process_sample(sample, eval_ty=eval_ty)
                    if model_loss is None:
                        continue
                    n_processed += 1
                    self._display_loss(model_loss, eval_ty=eval_ty)
        self.model = self.model.train()
        avg_time = (time.time() - start) / max(n_processed, 1)
        logger.info(f"[INFO] finished running on {eval_ty} set, average time / sample :", {avg_time})

    @staticmethod
    def _get_optim(model: nn.Module, config: GlobalConfig, lr_scale=1, weight_decay: Optional[float] = None):
        """Get optimizer"""
        weight_decay = default(weight_decay, config.weight_decay)
        if model is None:
            return None
        logger.info("initializing optimizer")
        if weight_decay <= 0:
            return Adam(model.parameters(), lr=config.lr * lr_scale)
        else:
            return AdamW(
                model.parameters(),
                lr=config.lr * lr_scale,
                weight_decay=weight_decay
            )

    def perturb_wts(self, state_dict, perturb_eps: float):

        if perturb_eps > 0:
            print("[INFO] PERTURBING MODEL WEIGHTS")
            new_state = deepcopy(state_dict)
            for name, param in state_dict.items():
                if param.numel() > 3:
                    scale = torch.norm(param) / (param.numel() ** (1 / 2))
                    new_state[name] = param + (torch.randn_like(param) * scale * perturb_eps)
            return new_state
        return state_dict

    def _init_model(self, model_path: Optional[str] = None):

        """Initialize model with dummy forward pass"""
        print("[INFO] initializing model...")
        print("[INFO] Num trainable parameters:", num_trainable_params(model=self.model))
        # potentially load from previous state
        if self.config.load_state or exists(model_path):
            self.load_state(path=default(model_path, self.config.model_load_path))
        self.clear_grad_dict()
        # make sure all modules in model are initialized via a dummy forward pass
        max_tries = 3
        dataset = self.datasets[EvalTy.TRAINING.value]
        for i in range(max_tries):
            print(f"[INFO] Running Dummy forward pass {i}")
            try:
                out = self.safe_eval(sample=dataset[i], eval_ty=EvalTy.TESTING, raise_exceptions=True)
            except Exception as e:  # noqa
                out = None
            if i > max_tries:
                # probably an error, so raise the exception
                print("[ERROR] unable to initialize model")
                assert out is None
                # raise the exception
                self.safe_eval(sample=dataset[i], eval_ty=EvalTy.TESTING, raise_exceptions=True)
            if out is None:
                continue
            else:
                break
        print("[INFO] Num trainable parameters (after init) :", num_trainable_params(model=self.model))
        print("[INFO] Successfully initialized model!")
        if exists(self.post_init_fn):
            print("[INFO] applying _pose_init fn")
            self.post_init_fn(self.model)

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from protein_learning.common.data.datasets.chain_dataset import ChainDataset
from protein_learning.common.data.data_types.model_output import ModelOutput
from protein_learning.common.data.data_types.model_input import ModelInput
from protein_learning.common.global_config import GlobalConfig
from protein_learning.models.model_abc.protein_model import ProteinModel
from protein_learning.training.train_utils import EvalTy
from protein_learning.training.trainer import Trainer


def to_npy(x):
    """Converts x to numpy.ndarray"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return [to_npy(i) for i in x]
    return x


class ModelStats(ABC):
    """Caclulate and store statistics for design model"""

    def __init__(
            self,
            config: GlobalConfig,
            model: ProteinModel,
            max_samples: int = -1,
            use_custom_forward: bool = False,
            raise_exceptions: bool = True,
            model_path: Optional[str]=None,
    ):
        self.model = model
        self.config = config
        self.max_samples = max_samples
        self.data = self._init_data()
        self.use_custom_forward = use_custom_forward
        self.raise_exceptions = raise_exceptions
        self.model_path = model_path,

    def evaluate_dataset(self, dataset: ChainDataset, n_cycles: int = 1, **kwargs):
        """Evaluate on all samples in dataset"""
        trainer = Trainer(
            config=self.config,
            model=self.model,
            train_data=dataset,
        )
        trainer.model = trainer.model.eval()
        for _ in range(n_cycles):
            data = dataset.get_dataloader(batch_size=min(len(dataset), 32), num_workers=4, shuffle=False)
            n_processed = 0
            print(f"[INFO] Beginning evaluation of {len(dataset)} targets...")
            with torch.no_grad():
                for idx, batch_data in enumerate(data):
                    print(f"[INFO] loaded batch of size {len(batch_data)}")
                    for sample in batch_data:
                        try:
                            if self.use_custom_forward:
                                model_out = self.custom_forward(trainer.model, sample, **kwargs)
                            else:
                                model_out = trainer.safe_eval(sample, eval_ty=EvalTy.VALIDATION, raise_exceptions=True)
                            model_out = model_out if not isinstance(model_out, List) else model_out[-1]
                            if n_processed % max(1, (len(dataset) // 10)) == 0:
                                print(f"[INFO] progress {1000 * (n_processed / len(dataset)) // 10}%, "
                                      f"-- {n_processed}/{len(dataset)}")
                            self.log_stats(model_out=model_out, model=trainer.model)
                            n_processed += 1
                        except Exception as e:
                            print(f"[ERROR] caught exception {e}... skipping")
                            if self.raise_exceptions:
                                raise e
                    if 0 < self.max_samples < n_processed:
                        break

    def custom_forward(self, model: ProteinModel, sample: ModelInput, **kwargs):
        """If a custom forward pass is needed, this can be implemented by
        subclass"""
        pass

    @abstractmethod
    def _init_data(self):
        pass

    @abstractmethod
    def log_stats(self, model_out: ModelOutput, model: ProteinModel):
        """Log statistics for single input sample"""
        pass

    def save(self, path):
        """Save statistics"""
        np.save(path, {k: list(map(to_npy, v)) for k, v in self.data.items()})

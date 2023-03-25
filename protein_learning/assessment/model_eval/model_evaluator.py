import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
from protein_learning.models.model_abc.protein_model import ProteinModel
from protein_learning.common.data.data_types.model_output import ModelOutput, ModelInput
from typing import Optional, Dict, Tuple, NamedTuple, List
from protein_learning.training.trainer import Trainer
from protein_learning.training.train_utils import EvalTy
from protein_learning.common.global_config import GlobalConfig
import numpy as np
import time
from protein_learning.assessment.model_eval.stats_handler import LossHandler
from protein_learning.common.helpers import exists, default
from enum import Enum
from protein_learning.common.data.data_types.protein import Protein
import os
from protein_learning.features.masking.masking_utils import get_chain_masks
import traceback
from protein_learning.common.data.datasets.utils import set_canonical_coords_n_masks
from torch.cuda.amp import autocast
from copy import deepcopy
import math
from protein_learning.common.rigids import Rigids

class StatsKeys(Enum):
    NSR = "nsr"
    RMSD = "rmsd"
    LDDT = "lddt"
    METADATA = "metadata"
    CONTACTS = "contacts"
    VIOLATION = "violation"
    MAE = "mae"
    DIST = "distance"


class StatsConfig(NamedTuple):
    get_nsr_stats: bool
    get_lddt_stats: bool
    get_violation_stats: bool
    get_rmsd_stats: bool
    get_contact_stats: bool
    get_mae_stats: bool
    get_distance_stats: bool


def random_name(prefix=""):
    return f"{prefix}{np.random.randint(0, int(1e16))}"


class ModelEvaluator:
    """Evaluate a model"""

    def __init__(
            self,
            stats_config: StatsConfig,
            global_config: Optional[GlobalConfig],
            model: Optional[ProteinModel],
            pdb_dir: Optional[str],
            dataset: Dataset,
            max_samples: int = -1,
            use_custom_forward: bool = False,
            model_path: Optional[str] = None,
            raise_exceptions: bool = True,
            pdb_only=False,
            use_amp = False,
            chunk_len=800,
    ):
        self.use_amp = use_amp
        self.stats_config = stats_config
        self.global_config = global_config
        self.model = model.eval()
        self.pdb_dir = pdb_dir
        self.dataset = dataset
        self.max_samples = max_samples if max_samples > 0 else len(dataset)
        self.use_custom_forward = use_custom_forward
        self.trainer, self.loss_fn = None, None
        if exists(self.model):
            self.trainer, self.model = self._init_model(model_path)
            self.loss_fn = self.model.get_loss_fn()
            assert exists(self.loss_fn)
        self.raise_exceptions = raise_exceptions
        self.handler = LossHandler(self.loss_fn)
        self.stats_data = []
        self.pdb_only = pdb_only
        self.use_amp = use_amp
        self.chunk_len=chunk_len

    def _init_model(self, model_path: str) -> Tuple[Trainer, ProteinModel]:
        trainer = Trainer(
            config=self.global_config,
            model=self.model,
            train_data=self.dataset,  # noqa
            load_optim = False,
        )
        trainer._init_model(model_path=model_path)  # noqa
        trainer.model = trainer.model.eval().to(self.global_config.device)
        return trainer, trainer.model

    def write_predicted_pdb(
            self,
            model_out: ModelOutput,
            seq: Optional[str],
    ) -> List[str]:
        """Write predicted pdbs"""
        
        pred_protein = make_predicted_protein(model_out, seq=seq)
        pdb_path = os.path.join(self.pdb_dir, model_out.decoy_protein.name)
        #pdb_path = random_name(pdb_path + "_")
        chains, names = ["A", "B"], []

        if model_out.native.n_chains > 1:
            for idx in range(model_out.native.n_chains):
                names.append(pdb_path + f"_{chains[idx]}.pdb")
                pred_protein.to_pdb(
                    path=names[-1],
                    chain_idx=idx,
                )
        names.append(pdb_path + f".pdb")
        print(f"writing predicted pdb to : {names[-1]}")
        pred_protein.to_pdb(
            path=os.path.join(names[-1]),
        )
        names.append(pdb_path + "_native.pdb")
        native_ptn = model_out.native.clone()
        """
        native_ptn.kabsch_align_coords(
            target=None,
            coords=model_out.predicted_coords[0, :, 1],
            atom_ty="CA",
            overwrite=True
        )
        """
        native_ptn.to_pdb(path=names[-1])
        return names

    def get_metadata(self, model_out: ModelOutput, pred_seq: Optional[str]):
        pred_pdbs = None
        if self.pdb_dir is not None:
            pred_pdbs = self.write_predicted_pdb(model_out, seq=pred_seq)
        _chain_masks = get_chain_masks(
            len(model_out.decoy_protein),
            chain_indices=model_out.decoy_protein.chain_indices
        )
        chain_masks, chain_pair_masks, inter_chain_mask = _chain_masks
        seq_mask, res_feat_mask, inter_chain_feat_mask = [None] * 3
        if exists(model_out.model_input.input_features):
            seq_mask = model_out.model_input.input_features.seq_mask
            res_feat_mask = model_out.model_input.input_features.feat_mask
            inter_chain_feat_mask = model_out.model_input.input_features.inter_chain_mask
        return dict(
            pred_pdbs=pred_pdbs,
            native_chain_pdbs=model_out.native.chain_names,
            decoy_chain_pdbs=model_out.native.chain_names,
            seq_mask=seq_mask,
            res_feat_mask=res_feat_mask,
            inter_chain_feat_mask=inter_chain_feat_mask,
            predicted_coords=model_out.predicted_coords,
            actual_coords=model_out.native.atom_coords,
            decoy_coords=model_out.decoy_protein.atom_coords,
            atom_tys=model_out.native.atom_tys,
            coord_mask=model_out.decoy_protein.atom_masks,
            pred_seq=default(pred_seq, model_out.decoy_protein.seq),
            native_seq=model_out.native.seq,
            chain_indices=model_out.decoy_protein.chain_indices,
            chain_masks=chain_masks,
            chain_pair_masks=chain_pair_masks,
            inter_chain_mask=inter_chain_mask,
            valid_res_mask=model_out.valid_residue_mask,
            chain_lens=getattr(model_out.decoy_protein, "chain_lens", None),
        )

    def custom_forward(self, model_input: ModelInput, **fwd_kwargs) -> ModelOutput:
        """Override if desired"""
        pass

    def log_stats(self, model_out: ModelOutput, inf_time=-1, extra=None) -> None:
        """get all statistics"""
        stats = {}
        if not self.pdb_only:
            stats.update(
                get_handler_stats(
                    handler=self.handler,
                    config=self.stats_config,
                    model_out=model_out
                )
            )
        pred_seq = None
        if StatsKeys.NSR.value in stats:
            pred_seq = stats[StatsKeys.NSR.value]["pred_seq"]
        stats[StatsKeys.METADATA.value] = self.get_metadata(model_out, pred_seq=pred_seq)
        stats[StatsKeys.METADATA.value]["inference_time"] = inf_time
        stats["extra"] = extra
        self.stats_data.append(_convert_dict(stats))

    def evaluate_from_pdb(self, device="cpu"):
        """Evaluate dataset"""
        data = self.dataset.get_dataloader(  # noqa
            batch_size=min(len(self.dataset), 8),
            num_workers=4,
            shuffle=False
        )
        print(f"[INFO] beginning evaluation of {len(self.dataset)} targets")
        for idx, batch_data in enumerate(data):
            for sample in batch_data:
                sample = sample.to(device)
                model_out = ModelOutput(
                    predicted_coords=sample.decoy.atom_coords.clone().unsqueeze(0),
                    scalar_output=None,
                    pair_output=None,
                    model_input=sample,
                )
                self.log_stats(model_out)
                print("finished", sample.decoy.name)

    def evaluate(self, n_replicas: int = 1, **fwd_kwargs):
        """Evaluate dataset"""
        for replica in range(n_replicas):
            if hasattr(self.dataset, 'replica'):
                self.dataset.replica = replica
            data = self.dataset.get_dataloader(  # noqa
                batch_size=min(len(self.dataset), 8),
                num_workers=4,
                shuffle=False
            )
            n_processed = 0
            print(f"[INFO] Beginning evaluation of {len(self.dataset)} targets...")
            with torch.no_grad():
                for idx, batch_data in enumerate(data):
                    start = time.time()
                    n_hit, n_miss = self._eval_batch(batch_data, **fwd_kwargs)
                    print(f"[INFO] finished evaluation batch in time"
                          f" {np.round(time.time() - start, 2)} seconds")
                    print(f"[INFO] number of samples processed successfully {n_hit}")
                    print(f"[INFO] number of samples missed {n_miss}")
                    n_processed += n_hit
                    if n_processed >= self.max_samples:
                        break

    def _eval_batch(self, batch, **fwd_kwargs):
        n_hit, n_miss = 0, 0
        inference_times, extra = [], [dict()]
        for sample in batch:
            try:
                start = time.time()
                if self.use_custom_forward:
                    model_out, extra, sts = self.custom_forward(
                        sample,
                        **fwd_kwargs
                    )
                else:
                    seq_len = len(sample.native.seq)
                    print(f"[INFO] : starting forward pass for {sample.native.name}, seq len: {seq_len}")
                    if seq_len>self.chunk_len:
                        model_out = self.chunk_inference(
                            sample,
                            max_len=self.chunk_len
                        )
                    else:
                        model_out = self.safe_eval(
                            sample,
                        )
                inference_times.append(time.time() - start)
                if model_out is None:
                    continue

            except Exception as e:
                nm = sample.native.name if (exists(sample) and exists(sample.native)) else None
                print(f"[ERROR] caught exception {e} running forward pass")
                print(traceback.format_exc())
                if self.raise_exceptions:
                    raise e
                print(f"Skipping {nm}")
                n_miss += 1
                continue

            try:
                model_out = model_out if isinstance(model_out, list) else [model_out]
                for m, e in zip(model_out, extra):
                    self.log_stats(m, inf_time=inference_times, extra=e)  # noqa
            except Exception as e:
                print(f"caught exception {e} gathering stats")
                if self.raise_exceptions:
                    raise e
                nm = sample.native.name if (exists(sample) and exists(sample.native)) else None
                print(f"Skipping {nm}")
                n_miss += 1
                continue
            n_hit += 1
            print(f"[INFO] : finished forward pass for {sample.native.name}, \
                inference time : {inference_times[-1]}"
             )
        return n_hit, n_miss

    def safe_eval(
        self,
        sample: ModelInput,
        crop = None,
    ) -> ModelOutput:
        """Evaluate model and (optionally) catch exceptions"""
        device, max_len = self.global_config.device,self.global_config.max_len
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                sample = sample.to(device).crop(max_len, bounds=crop)
                model_out = self.model.eval()(sample)
                return model_out#.detach()

    
    def chunk_inference(
            self,
            sample: ModelInput,
            max_len: int = 500,
    ):
        seq_len = len(sample.decoy.seq)
        crops, crop_offset = get_inference_crops(max_len,seq_len=seq_len)
        output = []
        for crop in crops:
            sample_copy = deepcopy(sample)
            output.append(self.safe_eval(sample=sample_copy,crop=crop))
        
        device = output[0].scalar_output.device
        scalar_out_shape = output[0].scalar_output.shape
        predicted_coords = torch.zeros(1,seq_len,37,3,device=device)
        scalar_output = torch.zeros(1,seq_len,scalar_out_shape[-1],device=device)
        quats,translations,angles=None,None,None
        if exists(output[0].angles):
            angles = torch.zeros(1,seq_len,*output[0].angles.shape[2:], device = device)
        if exists(output[0].pred_rigids):
            quats = torch.zeros(1,seq_len,*output[0].pred_rigids.quaternions.shape[2:], device = device)
            translations = torch.zeros(1,seq_len,*output[0].pred_rigids.translations.shape[2:], device = device)

        
        for i in range(len(crops)):
            offset = 0 if i==0 else crop_offset//2
            start, end = crops[i]
            curr_out = output[i]
            scalar_output[:,start+offset:end] = curr_out.scalar_output[:,offset:]
            predicted_coords[:,start+offset:end] = curr_out.predicted_coords[:,offset:]
            if exists(curr_out.pred_rigids):
                translations[:,start+offset:end] = curr_out.pred_rigids.translations[:,offset:]
                quats[:,start+offset:end] = curr_out.pred_rigids.quaternions[:,offset:]
            if exists(curr_out.angles):
                angles[:,start+offset:end] = curr_out.angles[:,offset:]


        #overwrite model out features
        return ModelOutput(
            scalar_output=scalar_output,
            predicted_coords=predicted_coords,
            pair_output=torch.ones(1),
            model_input=sample.to(device),
            extra = dict(
                pred_rigids = Rigids(quats,translations) if exists(quats) else None,
                chain_indices = sample.decoy.chain_indices,
                angles = angles,
                unnormalized_angles=angles
            )
        )
    """
    def _init_model(self, model: torch.nn.Module, model_path: Optional[str] = None):
        print("[INFO] initializing model...")
        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"[INFO]: Loading from path {model_path}")
        model.load_state_dict(checkpoint['model'], strict=True)
        return model
    """
"""Helper Methods"""

def get_inference_crops(
        max_len,
        seq_len,
    ):
    crop_offset = min(max_len//2,200)
    n_crops = math.ceil(seq_len/(max_len-crop_offset))
    crop_len = math.ceil(seq_len/n_crops) + crop_offset
    crops, start = [(0,max_len)], max_len-crop_offset
    while start < seq_len-crop_offset:
        crop = (start,min(seq_len,start+max_len))
        crops.append(crop)
        start += max_len - crop_offset
    crops = crops[:-1]
    crops.append((seq_len-max_len,seq_len))
    return crops, crop_offset


def _convert_tensor(x: torch.Tensor):
    x = x.squeeze(0).detach().cpu()
    if x.dtype in [torch.float32, torch.double, torch.float64, torch.float]:
        x = x.float()
    return x.numpy()


def _convert_value(value):
    if torch.is_tensor(value):
        return _convert_tensor(value)
    if isinstance(value, list):
        return [_convert_value(x) for x in value]
    if isinstance(value, tuple):
        return tuple([_convert_value(x) for x in value])
    return value


def _convert_dict(data, tab="") -> Dict:
    if isinstance(data, dict):
        converted_dict = {}
        for k, v in data.items():
            converted_dict[k] = _convert_dict(v, tab=tab + f"{k}    ")
        return converted_dict
    return _convert_value(data)


def get_handler_stats(
        handler: LossHandler,
        config: StatsConfig,
        model_out: ModelOutput
) -> Dict:
    """Get stats from loss handler"""
    stats = {}
    tmp = [
        (StatsKeys.NSR, handler.get_nsr_data, config.get_nsr_stats),
        (StatsKeys.RMSD, handler.get_rmsd_data, config.get_rmsd_stats),
        (StatsKeys.VIOLATION, handler.get_violation_data, config.get_violation_stats),
        (StatsKeys.LDDT, handler.get_lddt_data, config.get_lddt_stats),
        (StatsKeys.CONTACTS, handler.get_contact_data, config.get_contact_stats),
        (StatsKeys.MAE, handler.get_mae_data, config.get_mae_stats),
        (StatsKeys.DIST, handler.get_predicted_dist_data, config.get_distance_stats)
    ]
    for key, fn, use in tmp:
        try:
            if use:
                stats[key.value] = fn(model_out)
        except Exception as e:
            print(f"[ERROR] Getting {key.value} stats")
            raise e
    return stats




def make_predicted_protein(model_out: ModelOutput, seq: Optional[str] = None) -> Protein:
    """Constructs predicted protein"""
    coords = model_out.predicted_coords.squeeze(0)
    pred_protein = model_out.decoy_protein.from_coords_n_seq(coords,seq)
    pred_protein = set_canonical_coords_n_masks(pred_protein, overwrite=True)
    return pred_protein




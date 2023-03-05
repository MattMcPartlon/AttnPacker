import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
from protein_learning.models.fbb_design.train import (
    Train,
)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_parser():
    parser = ArgumentParser(description="Pack Side-Chains",  # noqa
                        epilog='',
                        formatter_class=ArgumentDefaultsHelpFormatter)
    add = lambda *args, **kwargs : parser.add_argument(*args, **kwargs)

    add("model")
    add("pdb_path")

    #I/O args
    add("--out_folder", default="./",help="directory to store results in")
    add("--pdb_folder", help="folder containing pdbs to run inference on "\
        "(will produce results for all pdbs in the referenced folder)")
    
    add("--fasta_folder", help="folder with ")
    



    flags = ["predict_confidence"]
    

class Inference(Train):

    def __init__(self):
        super().__init__()

    @property
    def load_optim(self):
        return False

    @property
    def global_override_eval(self):
        """kwargs to override in global model config"""
        return dict(batch_size=4,data_workers=4)

    def _model_override_eval(self):
        """Kwargs to override in model config for eval"""
        return dict(res_ty_corrupt_prob=0)
    
    def _add_extra_cmd_line_options_for_eval(self, parser):
        return parser

    @property
    def allow_missing_nn_modules(self):
        return False

if __name__ == "__main__":



    x = Inference()
    ty = "Training" if not x.do_eval else "Evaluation"
    print(f"[INFO] Beginning {ty} for Masked Design Model")
    x.run(detect_anomoly=False)
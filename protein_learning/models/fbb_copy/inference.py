from protein_learning.models.fbb_design.train import (
    Train,
)


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
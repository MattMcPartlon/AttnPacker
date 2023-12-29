# ProteinModel
Defines the abstract framework for creating and running a ProteinLearning Model

## protein_model.py
Abstract base class for all protein learning models

The forward pass takes a ModelInput object as input,
as well as any additional key work arguments

The forward pass looks as follows
(* lines indicate abstract method calls)

### Pseudocode for Forward Pass
```python
def forward(input : ModelInput):
    res_feats, pair_feats = embed(model_input)
    fwd_kwargs = get_forward_kwargs(
      model_input, res_feats, pair_feats,
    )
    output = self.model.forward(**fwd_kwargs)
    model_output = get_model_output(
      fwd_kwargs,
      **output,
    )
    self.finish_forward() # anything state maintenance
    return model_output # ModelOutput object
```


### Abstract Methods

```python
@abstractmethod
def get_forward_kwargs(
        self,
        model_input: ModelInput,
        residue_feats: Tensor,
        pair_feats: Tensor,
        **kwargs,
) -> Dict:
    """Get keyword arguments for protein model forward pass

    Params:
        model_input: moel input representing training sample
        residue_feats: (linearly projected) residue features from input embedding
        pair_feats: (linearly projected) pair features from input embedding

    Return: All input kwargs needed for structure module
    """
    pass
```
```python
def finish_forward(self) -> None:
    """Called at the very end of the forward pass
    This is the place to clear any state information you may have saved, e.g.
    """
    pass
```
```python
@abstractmethod
def get_model_output(self, fwd_output: Any, fwd_input: Dict, **kwargs)->ModelOutput:
    """Get Model output object from

    (1) output of model forward
    (2) input kwargs of forward pass
    """
```
```python
@abstractmethod
def compute_loss(self, output: ModelOutput, batch_idx : int, **kwargs) -> ModelLoss:
    """Compute the loss from the output"""
    pass
```

## StructureModel `structure_model.py`
In addition to the Standard model, a "structure prediction" abstract base class is also defined.
The main difference is that the structure abstract base class handles things like
recycling

### Pseudocode for Forward Pass
```python
def forward(input : ModelInput):
    init_res_feats, init_pair_feats = embed(model_input)
    
    for i in range(n_cycles)
      fwd_kwargs = get_forward_kwargs(
        model_input, res_feats, pair_feats,
      )
      output = self.model.forward(**fwd_kwargs)
      model_output = get_model_output(
        fwd_kwargs,
        **output,
      )
    self.finish_forward() # anything state maintenance
    return model_output # ModelOutput object
```


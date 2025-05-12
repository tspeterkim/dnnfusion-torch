# DNNFusion-Torch

Implementation of the operator-type-based fusion planner algorithm in DNNFusion, using Torch.fx.

## Install

```
pip install torch tabulate
```

## Evaluation

To run the algorithm for LeNet:
```
python lenet.py
```

To run torch.compile's Triton-based fusion:
```
TORCH_COMPILE_DEBUG=1 python eval.py
```

Pre-fusion and post fusion IRs are in `ir_pre_fusion.txt` and `ir_post_fusion.txt`, respectively.
For a more readable result (and the acctual generated Triton code), look at `output_code.py`.
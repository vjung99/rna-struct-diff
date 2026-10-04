#!/usr/bin/env python
"""Run network/predict.py on a CPU-only torch build.

SE3Transformer wraps its hot paths in torch.cuda.nvtx.range, which raises
"NV TX functions not installed" when torch has no CUDA build. The spans are pure
profiling scaffolding, so on CPU we replace them with a null context.

Usage: env/bin/python run_cpu.py -inputs R:query.afa -prefix out/model -model ... -db ...
"""
import contextlib
import os
import runpy
import sys

import torch

if not torch.cuda.is_available():
    import torch.cuda.nvtx as _nvtx

    _nvtx.range = _nvtx.range_push = _nvtx.range_pop = (
        lambda *args, **kwargs: contextlib.nullcontext()
    )

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "network"))
sys.argv[0] = os.path.join(HERE, "network", "predict.py")

runpy.run_path(sys.argv[0], run_name="__main__")

"""Pytest config for native MACA (cuda:0) — does NOT import torch_fl.

Run:
    pytest tests/integration/maca_native --confcutdir=tests/integration/maca_native -v -s \\
        --model /home/hongzw/Qwen3-0.6B --max-new-tokens 16 --warmup-rounds 1
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--model", default="/home/hongzw/Qwen3-0.6B")
    parser.addoption("--max-new-tokens", type=int, default=16)
    parser.addoption("--warmup-rounds", type=int, default=1)

#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname $0)/../benchmarks/dynamic_cityscapes_finetune_evaluate.py
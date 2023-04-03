# GAIA-seg
An AutoML toolbox specialized in semantic segmentation. 
# Install

  ## requirements:
  torch 1.9.0+
  
  gaiavision
  
  mmcv-full 1.3.9

# Command
  ## Supernet training
  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_local.sh config.py 8
  ```

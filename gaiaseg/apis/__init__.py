from .inference import (inference_segmentor,
                        init_segmentor, show_result_pyplot)
from .train import train_segmentor
from .test import multi_gpu_test, single_gpu_test
__all__ = [
    'train_segmentor', 'init_segmentor',
     'inference_segmentor', 'show_result_pyplot',
    'multi_gpu_test', 'single_gpu_test'
]

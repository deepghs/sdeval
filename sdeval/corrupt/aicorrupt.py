import json
from functools import lru_cache
from typing import Tuple, Optional, Mapping

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

from ..utils import ImagesTyping, load_images

_DEFAULT_MODEL_NAME = 'caformer_s36_v0_focal'


@lru_cache()
def _open_anime_aicop_model(model_name):
    return open_onnx_model(hf_hub_download(
        f'deepghs/ai_image_corrupted',
        f'{model_name}/model.onnx',
    ))


@lru_cache()
def _open_anime_aicop_meta(model_name):
    with open(hf_hub_download(
            f'deepghs/ai_image_corrupted',
            f'{model_name}/meta.json',
    ), 'r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache()
def _open_anime_aicop_labels(model_name):
    return _open_anime_aicop_meta(model_name)['labels']


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


def get_ai_corrupted(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Mapping[str, float]:
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_aicop_model(model_name).run(['output'], {'input': input_})
    return dict(zip(_open_anime_aicop_labels(model_name), output[0].tolist()))


class AICorruptMetrics:
    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME):
        self._model_name = model_name

    def corrupt_score(self, images: ImagesTyping):
        image_list = load_images(images)
        if not image_list:
            raise FileNotFoundError(f'Images for calculating AI corrupt score not provided - {images}.')

        scores = np.array([get_ai_corrupted(image, model_name=self._model_name)['corrupted'] for image in image_list])
        return scores.mean().item()

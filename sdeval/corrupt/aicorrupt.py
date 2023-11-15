"""
Overview:
    AI image corrupt evaluation metrics.
"""
import json
from functools import lru_cache
from typing import Tuple, Optional, Mapping

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from imgutils.data import rgb_encode, ImageTyping, load_image
from imgutils.utils import open_onnx_model

from ..utils import ImagesTyping, load_images, tqdm

_DEFAULT_MODEL_NAME = 'caformer_s36_v0_focal'


@lru_cache()
def _open_anime_aicop_model(model_name: str):
    """
    Open the AI image corrupted detection model.

    This function downloads and opens the AI image corrupted detection model specified by the given model name using Hugging Face Hub.

    :param model_name: The name of the AI image corrupted detection model.
    :type model_name: str

    :return: The opened AI image corrupted detection model.
    :rtype: Model
    """
    return open_onnx_model(hf_hub_download(
        f'deepghs/ai_image_corrupted',
        f'{model_name}/model.onnx',
    ))


@lru_cache()
def _open_anime_aicop_meta(model_name: str):
    """
    Open the meta information of the AI image corrupted detection model.

    This function downloads and opens the meta information of the AI image corrupted detection model specified by the given model name using Hugging Face Hub.

    :param model_name: The name of the AI image corrupted detection model.
    :type model_name: str

    :return: The opened meta information of the AI image corrupted detection model.
    :rtype: dict
    """
    with open(hf_hub_download(
            f'deepghs/ai_image_corrupted',
            f'{model_name}/meta.json',
    ), 'r', encoding='utf-8') as f:
        return json.load(f)


@lru_cache()
def _open_anime_aicop_labels(model_name: str):
    """
    Open the labels of the AI image corrupted detection model.

    This function opens the labels of the AI image corrupted detection model specified by the given model name.

    :param model_name: The name of the AI image corrupted detection model.
    :type model_name: str

    :return: The labels of the AI image corrupted detection model.
    :rtype: List[str]
    """
    return _open_anime_aicop_meta(model_name)['labels']


def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    """
    Encode the image for AI image corrupted detection.

    This function resizes and encodes the image for AI image corrupted detection.

    :param image: The input image.
    :type image: Image.Image
    :param size: The target size for encoding. Default is (384, 384).
    :type size: Tuple[int, int]
    :param normalize: The normalization parameters. Default is (0.5, 0.5).
    :type normalize: Optional[Tuple[float, float]]

    :return: The encoded image data.
    :rtype: np.ndarray
    """
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')

    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std

    return data.astype(np.float32)


def get_ai_corrupted(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME) -> Mapping[str, float]:
    """
    Get AI image corrupted detection scores for an image.

    This function calculates AI image corrupted detection scores for a given image using the specified model.

    :param image: The input image.
    :type image: ImageTyping
    :param model_name: The name of the AI image corrupted detection model. Default is 'caformer_s36_v0_focal'.
    :type model_name: str

    :return: A dictionary containing the corrupted score.
    :rtype: Mapping[str, float]
    """
    image = load_image(image, force_background='white', mode='RGB')
    input_ = _img_encode(image)[None, ...]
    output, = _open_anime_aicop_model(model_name).run(['output'], {'input': input_})
    return dict(zip(_open_anime_aicop_labels(model_name), output[0].tolist()))


class AICorruptMetrics:
    """
    Class for calculating an AI image corruptness score.

    The `AICorruptMetrics` class allows you to calculate an AI image corruptness score using the AI image corrupted detection model.

    :param model_name: The name of the AI image corrupted detection model. Default is 'caformer_s36_v0_focal'.
    :type model_name: str
    :param silent: If True, suppresses progress bars and additional output during calculation.
    :type silent: bool
    :param tqdm_desc: Description for the tqdm progress bar during calculation.
    :type tqdm_desc: str
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME,
                 silent: bool = False, tqdm_desc: str = None):
        self._model_name = model_name
        self.silent = silent
        self.tqdm_desc = tqdm_desc or self.__class__.__name__

    def score(self, images: ImagesTyping, silent: bool = None):
        """
        Calculate the AI image corruptness score for a set of images.

        This method calculates the AI image corruptness score for a set of input images using the AI image corrupted detection model.

        :param images: The set of input images for calculating the AI image corruptness score.
        :type images: ImagesTyping
        :param silent: If True, suppresses progress bars and additional output during calculation.
        :type silent: bool

        :return: The AI image corruptness score.
        :rtype: float
        """
        image_list = load_images(images)
        if not image_list:
            raise FileNotFoundError(f'Images for calculating AI corrupt score not provided - {images}.')

        scores = np.array([
            get_ai_corrupted(image, model_name=self._model_name)['corrupted']
            for image in tqdm(image_list, silent=self.silent if silent is None else silent, desc=self.tqdm_desc)
        ])
        return 1.0 - scores.mean().item()

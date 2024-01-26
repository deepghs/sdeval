"""
Overview:
    CCIP-based metrics for anime character training.

    See `imgutils.metrics.ccip <https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html>`_ for more information.
"""
import warnings
from typing import List, Optional, Literal, Union

import numpy as np
from PIL import Image
from imgutils.metrics import ccip_extract_feature, ccip_default_threshold, ccip_batch_differences

from ..utils import load_images, ImagesTyping, tqdm

_DEFAULT_CCIP_MODEL = 'ccip-caformer-24-randaug-pruned'


class CCIPMetrics:
    """
    Class for calculating similarity scores between images using the CCIP (Content-Consistent Image Pairwise) metric.

    The `CCIPMetrics` class allows you to calculate the similarity score between a set of images and a reference dataset using the CCIP metric.

    :param images: The reference dataset of images for initializing CCIP metrics.
    :type images: ImagesTyping
    :param feats: Feature data of given character, should be (B, 768). When assigned, ``images`` argument will be ignored.
    :type feats: Optional[np.ndarray]
    :param model: The CCIP model to use for feature extraction. Default is 'ccip-caformer-24-randaug-pruned'.
    :type model: str
    :param threshold: The threshold for the CCIP metric. If not provided, the default threshold for the chosen model is used.
    :type threshold: Optional[float]
    :param silent: If True, suppresses progress bars and additional output during initialization and calculation.
    :type silent: bool
    :param tqdm_desc: Description for the tqdm progress bar during initialization and calculation.
    :type tqdm_desc: str
    """

    def __init__(self, images: ImagesTyping, feats: Optional[np.ndarray] = None, model: str = _DEFAULT_CCIP_MODEL,
                 threshold: Optional[float] = None, silent: bool = False, tqdm_desc: str = None):
        self.silent = silent
        self.tqdm_desc = tqdm_desc or self.__class__.__name__
        self._ccip_model = model
        self._threshold = ccip_default_threshold(self._ccip_model) if threshold is None else threshold

        if feats is None:
            image_list: List[Image.Image] = load_images(images)
            if not image_list:
                raise FileNotFoundError(f'Images for initializing CCIP metrics not provided - {images}.')
            self._features = [
                ccip_extract_feature(img, model=self._ccip_model)
                for img in tqdm(image_list, silent=self.silent, desc=f'{self.tqdm_desc} Initializing')
            ]

        else:
            if images:
                warnings.warn(f'Binary features assigned, images {images!r} will be ignored.')
            if len(feats.shape) != 2 or feats.shape[-1] != 768:
                raise ValueError(f'Feature shape should be (B, 768), but {feats.shape!r} found actually.')
            self._features = list(feats)

    def score(self, images: ImagesTyping, silent: bool = None,
              mode: Literal['mean', 'seq'] = 'mean') -> Union[float, np.ndarray]:
        """
        Calculate the similarity score between the reference dataset and a set of input images.

        This method calculates the similarity score between the reference dataset (used for initialization) and a set of input images using the CCIP metric.

        :param images: The set of input images for calculating CCIP metrics.
        :type images: ImagesTyping
        :param silent: If True, suppresses progress bars and additional output during calculation.
        :type silent: bool
        :param mode: Mode of the return value. Return a float value when ``mean`` is assigned,
                    return a numpy array when ``seq`` is assigned. Default is ``mean``.
        :type mode: Literal['mean', 'seq']

        :return: The similarity score between the reference dataset and the input images.
        :rtype: Union[float, np.ndarray]
        """
        image_list: List[Image.Image] = load_images(images)
        if not image_list:
            raise FileNotFoundError(f'Images for calculating CCIP metrics not provided - {images}.')

        _features = [
            ccip_extract_feature(img, model=self._ccip_model)
            for img in tqdm(image_list, silent=self.silent if silent is None else silent,
                            desc=f'{self.tqdm_desc} Calculating')
        ]

        diffs = ccip_batch_differences([*self._features, *_features])
        matrix = diffs[:len(self._features), len(self._features):]
        seq = (matrix < self._threshold).mean(axis=0)
        assert seq.shape == (len(_features),)

        if mode == 'seq':
            return seq
        else:
            return seq.mean().item()

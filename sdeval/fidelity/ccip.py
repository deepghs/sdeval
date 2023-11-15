"""
Overview:
    CCIP-based metrics for anime character training.

    See `imgutils.metrics.ccip <https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html>`_ for more information.
"""
from typing import List, Optional

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
    :param model: The CCIP model to use for feature extraction. Default is 'ccip-caformer-24-randaug-pruned'.
    :type model: str
    :param threshold: The threshold for the CCIP metric. If not provided, the default threshold for the chosen model is used.
    :type threshold: Optional[float]
    :param silent: If True, suppresses progress bars and additional output during initialization and calculation.
    :type silent: bool
    :param tqdm_desc: Description for the tqdm progress bar during initialization and calculation.
    :type tqdm_desc: str
    """

    def __init__(self, images: ImagesTyping, model: str = _DEFAULT_CCIP_MODEL,
                 threshold: Optional[float] = None, silent: bool = False, tqdm_desc: str = None):
        image_list: List[Image.Image] = load_images(images)
        if not image_list:
            raise FileNotFoundError(f'Images for initializing CCIP metrics not provided - {images}.')

        self.silent = silent
        self.tqdm_desc = tqdm_desc or self.__class__.__name__
        self._ccip_model = model
        self._features = [
            ccip_extract_feature(img, model=self._ccip_model)
            for img in tqdm(image_list, silent=self.silent, desc=f'{self.tqdm_desc} Initializing')
        ]
        self._threshold = ccip_default_threshold(self._ccip_model) if threshold is None else threshold

    def score(self, images: ImagesTyping, silent: bool = None) -> float:
        """
        Calculate the similarity score between the reference dataset and a set of input images.

        This method calculates the similarity score between the reference dataset (used for initialization) and a set of input images using the CCIP metric.

        :param images: The set of input images for calculating CCIP metrics.
        :type images: ImagesTyping
        :param silent: If True, suppresses progress bars and additional output during calculation.
        :type silent: bool

        :return: The similarity score between the reference dataset and the input images.
        :rtype: float
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

        return (matrix < self._threshold).mean().item()

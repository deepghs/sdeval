from typing import List, Optional, Tuple

from PIL import Image
from imgutils.metrics import ccip_extract_feature, ccip_default_threshold, ccip_batch_differences

from ..utils import load_images, ImagesTyping, tqdm

_DEFAULT_CCIP_MODEL = 'ccip-caformer-24-randaug-pruned'


class CCIPMetrics:
    def __init__(self, images: ImagesTyping, model: str = _DEFAULT_CCIP_MODEL,
                 threshold: Optional[float] = None, silent: bool = False):
        image_list: List[Image.Image] = load_images(images)
        if not image_list:
            raise FileNotFoundError(f'Images for initializing CCIP metrics not provided - {images}.')

        self.silent = silent
        self._ccip_model = model
        self._features = [
            ccip_extract_feature(img, model=self._ccip_model)
            for img in tqdm(image_list, silent=self.silent)
        ]
        self._threshold = ccip_default_threshold(self._ccip_model) if threshold is None else threshold

    def diff_score(self, images: ImagesTyping, silent: bool = None) -> Tuple[float, float]:
        image_list: List[Image.Image] = load_images(images)
        if not image_list:
            raise FileNotFoundError(f'Images for calculating CCIP metrics not provided - {images}.')

        _features = [
            ccip_extract_feature(img, model=self._ccip_model)
            for img in tqdm(image_list, silent=self.silent if silent is None else silent)
        ]

        diffs = ccip_batch_differences([*self._features, *_features])
        matrix = diffs[:len(self._features), len(self._features):]

        return matrix.mean().item(), (matrix < self._threshold).mean().item()

    def diff(self, images: ImagesTyping, silent: bool = None) -> float:
        diff, score = self.diff_score(images, silent)
        return diff

    def score(self, images: ImagesTyping, silent: bool = None) -> float:
        diff, score = self.diff_score(images, silent)
        return score

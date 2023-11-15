from typing import List, Optional

from PIL import Image
from imgutils.metrics import ccip_extract_feature, ccip_default_threshold, ccip_batch_differences

from ..utils import load_images, ImagesTyping, tqdm

_DEFAULT_CCIP_MODEL = 'ccip-caformer-24-randaug-pruned'


class CCIPMetrics:
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

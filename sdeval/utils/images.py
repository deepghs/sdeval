import os.path
from typing import List, Iterator, Union

from PIL import UnidentifiedImageError, Image
from imgutils.data import load_image

ImagesTyping = Union[Image.Image, str, List[Union[Image.Image, str]]]


def _yield_images(images: ImagesTyping) -> Iterator[Image.Image]:
    if isinstance(images, list):
        for item in images:
            yield from _yield_images(item)
    elif isinstance(images, str) and os.path.isdir(images):
        for root, dirs, files in os.walk(images):
            for file in files:
                yield from _yield_images(os.path.join(images, root, file))
    else:
        try:
            image = load_image(images)
            image.load()
            yield image
        except UnidentifiedImageError:
            pass


def load_images(images: ImagesTyping) -> List[Image.Image]:
    return list(_yield_images(images))

import os.path
from typing import List, Iterator, Union

from PIL import UnidentifiedImageError, Image
from imgutils.data import load_image

ImagesTyping = Union[Image.Image, str, List[Union[Image.Image, str]]]


def _yield_images(images: ImagesTyping) -> Iterator[Image.Image]:
    """
    Yield PIL.Image objects from various sources.

    This internal function yields PIL.Image objects from a variety of sources, including PIL.Image objects, file paths, and lists of images. It supports recursive loading of images from directories.

    :param images: An image or a list of images (PIL.Image, file paths, or a combination).
    :type images: ImagesTyping

    :return: An iterator yielding PIL.Image objects.
    :rtype: Iterator[Image.Image]
    """
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
    """
    Load multiple PIL.Image objects from various sources.

    This function loads multiple PIL.Image objects from a variety of sources, including PIL.Image objects, file paths, and lists of images. It supports recursive loading of images from directories.

    :param images: An image or a list of images (PIL.Image, file paths, or a combination).
    :type images: ImagesTyping

    :return: A list of PIL.Image objects.
    :rtype: List[Image.Image]
    """
    return list(_yield_images(images))

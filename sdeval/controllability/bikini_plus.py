"""
Overview:
    Bikini plus score.
"""
import json
import os
import re
import warnings
import weakref
from functools import lru_cache, partial
from queue import Queue
from typing import Optional, Tuple, List, Iterator, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from hbutils.string import singular_form
from huggingface_hub import hf_hub_download
from imgutils.data import load_image
from imgutils.sd import get_sdmeta_from_image
from imgutils.tagging import get_deepdanbooru_tags, get_wd14_tags, get_mldanbooru_tags

from ..utils import tqdm

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

_NOTHING = object()


class ACNode:
    def __init__(self, segments: Tuple[str, ...],
                 value=_NOTHING, fail_ref: 'ACNode' = None, is_root: bool = False):
        """
        AC (Aho-Corasick) Node class for fast tag matching.

        :param segments: Tuple of segments representing a path in the AC trie.
        :type segments: Tuple[str, ...]
        :param value: Value associated with the node, if any.
        :param fail_ref: Reference to the fail node.
        :type fail_ref: ACNode
        :param is_root: Indicates if the node is the root of the AC trie.
        :type is_root: bool
        """
        self.segments = segments
        if value is _NOTHING:
            self.has_value, self.value = False, None
        else:
            self.has_value, self.value = True, value

        self.children = {}
        self._fail_ref: Optional[weakref.ref] = None
        if is_root:
            self.fail = self
        else:
            if fail_ref is None:
                raise ValueError('Fail reference not given for non-root node.')  # pragma: no cover
            self.fail = fail_ref

    @property
    def fail(self) -> Optional['ACNode']:
        """
        Get the fail node reference.

        :return: Reference to the fail node.
        :rtype: Optional[ACNode]
        """
        if self._fail_ref is None:
            return None  # pragma: no cover
        else:
            return self._fail_ref()

    @fail.setter
    def fail(self, node: 'ACNode'):
        """
        Set the fail node.

        :param node: The fail node.
        :type node: ACNode
        """
        self._fail_ref = weakref.ref(node)


@lru_cache()
def _tag_list(tagger_name: str):
    """
    Get the tag list for a given tagger.

    :param tagger_name: Name of the tagger.
    :type tagger_name: str

    :return: List of tags.
    :rtype: List[dict]
    """
    with open(hf_hub_download(
            'deepghs/tagger_vocabs',
            filename=f'{tagger_name}/tags.json',
            repo_type='dataset'
    ), 'r', encoding='utf-8') as f:
        return json.load(f)


def _tokenize(text: str):
    """
    Tokenize the given text.

    :param text: Input text.
    :type text: str

    :return: List of tokens.
    :rtype: List[str]
    """
    return [singular_form(word) for word in re.split(r'[\W_]+', text) if word]


class TaggerACModel:
    def __init__(self, tagger_name: str):
        """
        Aho-Corasick (AC) model for fast tag matching.

        :param tagger_name: Name of the tagger.
        :type tagger_name: str
        """
        self._root_node = ACNode((), is_root=True)
        counts = []
        for tag in _tag_list(tagger_name):
            words_list = tag['words']
            for words in words_list:
                node = self._root_node
                for i, word in enumerate(words, start=1):
                    if word not in node.children:
                        node.children[word] = ACNode(
                            segments=tuple(words[:i]),
                            value=tag if i == len(words) else _NOTHING,
                            fail_ref=self._root_node,
                        )
                    node = node.children[word]
            counts.append(tag['count'])

        queue = Queue()
        queue.put(self._root_node)
        while not queue.empty():
            current_node: ACNode = queue.get()
            for key, child in current_node.children.items():
                if current_node is not self._root_node and key in current_node.fail.children:
                    child.fail = current_node.fail.children[key]

                queue.put(child)

        counts = np.array(counts)
        self._counts = np.log(counts[counts > 0])
        self._mean_count = np.percentile(self._counts, 75).item()

    def extract_tags_from_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract tags from the given text.

        :param text: Input text.
        :type text: str

        :return: List of tuples containing tags and their values.
        :rtype: List[Tuple[str, float]]
        """
        tokens = _tokenize(text)
        _exist_names = set()
        retval = []
        node = self._root_node
        for token in tokens:
            while node is not self._root_node:
                if token not in node.children:
                    node = node.fail
                else:
                    break

            if token in node.children:
                node = node.children[token]
                cur_node = node
                while cur_node is not self._root_node:
                    if cur_node.has_value:
                        tag_info = cur_node.value
                        if tag_info['name'] not in _exist_names:
                            value = np.log(tag_info['count']).item() - self._mean_count
                            retval.append((tag_info['name'], value))
                            _exist_names.add(tag_info['name'])
                    cur_node = cur_node.fail

            else:
                node = self._root_node

        return sorted(retval, key=lambda x: (-x[1], x[0]))

    def get_tag(self, tag_text):
        """
        Get the tag from the given tag text.

        :param tag_text: Input tag text.
        :type tag_text: str

        :return: Tag name.
        :rtype: str
        """
        tokens = _tokenize(tag_text)
        node = self._root_node
        for token in tokens:
            if token not in node.children:
                raise ValueError(f'Unknown tag {tag_text!r}.')

            node = node.children[token]

        if node.has_value:
            return node.value['name']
        else:
            raise ValueError(f'Unknown tag {tag_text!r}.')


def _deepdanbooru_tagging(image: Image.Image, use_real_name: bool = False,
                          general_threshold: float = 0.0, character_threshold: float = 0.0, **kwargs):
    _ = kwargs
    _, features, characters = get_deepdanbooru_tags(image, use_real_name, general_threshold, character_threshold)
    return {**features, **characters}


def _wd14_tagging(image: Image.Image, model_name: str,
                  general_threshold: float = 0.0, character_threshold: float = 0.0, **kwargs):
    _ = kwargs
    _, features, characters = get_wd14_tags(image, model_name, general_threshold, character_threshold)
    return {**features, **characters}


def _mldanbooru_tagging(image: Image.Image, use_real_name: bool = False, general_threshold: float = 0.0, **kwargs):
    _ = kwargs
    features = get_mldanbooru_tags(image, use_real_name, general_threshold)
    return features


_WD14_TAGGER_MODELS = {
    "wd14_swinv2": "wd-v1-4-swinv2-tagger-v2",
    "wd14_convnext": "wd-v1-4-convnext-tagger-v2",
    "wd14_convnextv2": "wd-v1-4-convnextv2-tagger-v2",
    "wd14_vit": "wd-v1-4-vit-tagger-v2",
    "wd14_moat": "wd-v1-4-moat-tagger-v2",
}
_TAGGING_METHODS = {
    'deepdanbooru': _deepdanbooru_tagging,
    'wd14_vit': partial(_wd14_tagging, model_name='ViT'),
    'wd14_convnext': partial(_wd14_tagging, model_name='ConvNext'),
    'wd14_convnextv2': partial(_wd14_tagging, model_name='ConvNextV2'),
    'wd14_swinv2': partial(_wd14_tagging, model_name='SwinV2'),
    'wd14_moat': partial(_wd14_tagging, model_name='MOAT'),
    'mldanbooru': _mldanbooru_tagging,
}

TaggingMethodTyping = Literal[
    'deepdanbooru', 'mldanbooru',
    'wd14_vit', 'wd14_convnext', 'wd14_convnextv2', 'wd14_swinv2', 'wd14_moat',
]

PromptedImageTyping = Union[
    Tuple[Image.Image, str, str], Tuple[Image.Image, str], Image.Image,
    Tuple[str, str, str], Tuple[str, str], str
]
PromptedImagesTyping = Union[PromptedImageTyping, List[PromptedImageTyping]]


def _yield_images(images: PromptedImagesTyping) -> Iterator[Tuple[Image.Image, str, str]]:
    """
    Yield images along with prompts and negative prompts.

    :param images: Input images with prompts and negative prompts.
    :type images: PromptedImagesTyping

    :return: Iterator of image, prompt, and negative prompt tuples.
    :rtype: Iterator[Tuple[Image.Image, str, str]]
    """
    if isinstance(images, list):
        for item in images:
            yield from _yield_images(item)
    elif isinstance(images, str) and os.path.isdir(images):
        for root, dirs, files in os.walk(images):
            for file in files:
                yield from _yield_images(os.path.join(images, root, file))
    else:
        if isinstance(images, tuple):
            if len(images) == 2:
                (img, prompt), neg_prompt = images, ''
            elif len(images) == 3:
                img, prompt, neg_prompt = images
            else:
                raise TypeError(f'Unknown tuple for prompted image - {images!r}.')
            img = load_image(img)
            img.load()

        else:
            try:
                img = load_image(images, force_background=None, mode=None)
                img.load()

                sdmeta = get_sdmeta_from_image(img)
                if sdmeta is None:
                    prompt, neg_prompt = '', ''
                else:
                    prompt, neg_prompt = sdmeta.prompt, sdmeta.neg_prompt
            except UnidentifiedImageError:
                return

        yield img, prompt, neg_prompt


class BikiniPlusMetrics:
    """
    Class for evaluating the appropriateness of AI-generated images based on prompts and taggers.

    The `BikiniPlusMetrics` class assesses the compatibility of AI-generated images with given prompts using taggers.

    :param tagger: The tagging method to use. Default is 'wd14_convnextv2'.
    :type tagger: TaggingMethodTyping
    :param tagger_cfgs: Optional configuration parameters for the chosen tagger. Default is None.
    :type tagger_cfgs: Optional[dict]
    :param base_num: Base number for weighting prompt tags. Default is 1.5.
    :type base_num: float
    :param tag_blacklist: Optional list of tags to exclude from evaluation. Default is None.
    :type tag_blacklist: Optional[List[str]]
    :param silent: If True, suppresses progress bars and additional output during calculation. Default is False.
    :type silent: bool
    """

    def __init__(self, tagger: TaggingMethodTyping = 'wd14_convnextv2',
                 tagger_cfgs: Optional[dict] = None, base_num: float = 1.5,
                 tag_blacklist: Optional[List[str]] = None, silent: bool = False):
        self.tagger = tagger
        self._tagger_func = partial(_TAGGING_METHODS[tagger], **(tagger_cfgs or {}))
        self._ac_model = TaggerACModel(_WD14_TAGGER_MODELS.get(tagger, tagger))
        self._base_num = base_num

        self._tag_blacklist_set = set()
        _unknown_blacklist_tags = set()
        for tag in (tag_blacklist or []):
            try:
                self._tag_blacklist_set.add(self._ac_model.get_tag(tag))
            except ValueError:
                _unknown_blacklist_tags.add(tag)
        if _unknown_blacklist_tags:
            warnings.warn(f'Unknown tags for blacklist: {sorted(_unknown_blacklist_tags)}.')
        self.silent = silent

    def _calculate_one_image(self, img: Image.Image, prompt: str, neg_prompt: str):
        """
        Calculate the bikini plus score for a single image.

        This method computes the bikini plus score for a single image based on the provided prompts.

        :param img: The input image.
        :type img: Image.Image
        :param prompt: The positive prompt for evaluation.
        :type prompt: str
        :param neg_prompt: The negative prompt for evaluation.
        :type neg_prompt: str

        :return: The calculated bikini plus score for the image.
        :rtype: float
        """
        prompt_tags = self._ac_model.extract_tags_from_text(prompt)
        prompt_tags = [(tag, value) for tag, value in prompt_tags if tag not in self._tag_blacklist_set]
        neg_prompt_tags = self._ac_model.extract_tags_from_text(neg_prompt)
        neg_prompt_tags = [(tag, value) for tag, value in neg_prompt_tags if tag not in self._tag_blacklist_set]
        tagged_tags = self._tagger_func(img)

        if not prompt_tags and not neg_prompt_tags:
            return 1.0

        vs = np.array([
            *(tagged_tags.get(tag, 0.0) for tag, value in prompt_tags),
            *((1.0 - tagged_tags.get(tag, 0.0)) for tag, value in neg_prompt_tags),
        ])
        ws = np.array([
            *((self._base_num ** value) for tag, value in prompt_tags),
            *((self._base_num ** value) for tag, value in neg_prompt_tags),
        ])
        return ((vs * ws).sum() / ws.sum()).item()

    def score(self, images: PromptedImagesTyping, silent: bool = False):
        """
        Calculate the average bikini plus score for a set of images.

        This method computes the average bikini plus score for a set of images based on the provided prompts.

        :param images: The set of images with associated positive and negative prompts.
        :type images: PromptedImagesTyping
        :param silent: If True, suppresses progress bars and additional output during calculation. Default is False.
        :type silent: bool

        :return: The average bikini plus score for the set of images.
        :rtype: float
        """
        image_list = list(_yield_images(images))
        if not image_list:
            raise FileNotFoundError(f'Images for calculating bikini plus score not provided - {images}.')

        return np.array([
            self._calculate_one_image(img, prompt, neg_prompt)
            for img, prompt, neg_prompt in tqdm(image_list, silent=self.silent if silent is None else silent)
        ]).mean().item()

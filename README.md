# sdeval

[![PyPI](https://img.shields.io/pypi/v/sdeval)](https://pypi.org/project/sdeval/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sdeval)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/8de0d1fd731d9bf1b984a28a6ed21494/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/8de0d1fd731d9bf1b984a28a6ed21494/raw/comments.json)

[![Code Test](https://github.com/deepghs/sdeval/workflows/Code%20Test/badge.svg)](https://github.com/deepghs/sdeval/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/deepghs/sdeval/workflows/Package%20Release/badge.svg)](https://github.com/deepghs/sdeval/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/deepghs/sdeval/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/deepghs/sdeval)

![GitHub Org's stars](https://img.shields.io/github/stars/deepghs)
[![GitHub stars](https://img.shields.io/github/stars/deepghs/sdeval)](https://github.com/deepghs/sdeval/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/deepghs/sdeval)](https://github.com/deepghs/sdeval/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/deepghs/sdeval)
[![GitHub issues](https://img.shields.io/github/issues/deepghs/sdeval)](https://github.com/deepghs/sdeval/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/deepghs/sdeval)](https://github.com/deepghs/sdeval/pulls)
[![Contributors](https://img.shields.io/github/contributors/deepghs/sdeval)](https://github.com/deepghs/sdeval/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/deepghs/sdeval)](https://github.com/deepghs/sdeval/blob/master/LICENSE)

Evaluation for stable diffusion model training

## Installation

You can simply install it with `pip` command line from the official PyPI site.

```shell
pip install sdeval
```

If your operating environment includes a available GPU, you can use the following installation command to achieve higher
performance:

```shell
pip install sdeval[gpu]
```

For more information about installation, you can refer
to [Installation](https://deepghs.github.io/sdeval/main/tutorials/installation/index.html).

## Quick Start

### CCIP Metrics

```python
from sdeval.fidelity import CCIPMetrics

ccip = CCIPMetrics(images='/path/of/character/dataset')

# ccip score of one image
print(ccip.score('/path/of/one/image'))

# ccip score of a directory of images
print(ccip.score('/directory/of/images'))

```

### Bikini Plus Metrics

```python
from sdeval.controllability import BikiniPlusMetrics

# build bikini plus score metrics
bp = BikiniPlusMetrics(
    tag_blacklist=[
        'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
        'breasts', 'grey_hair', 'medium_breasts'
    ]
)

# bp score of one image
# the image should contain a1111 webui's metadata of prompts
print(bp.score('/path/of/one/image'))

# bp score of a directory of images
# the images should contain a1111 webui's metadata of prompts
print(bp.score('/directory/of/images'))

```

### AI Corrupt Metrics

```python
from sdeval.corrupt import AICorruptMetrics

# build metrics
metrics = AICorruptMetrics()

# get ai corrupt score for one image file
print(metrics.score('/path/of/one/image'))

# get ai corrupt score of a directory of image files
print(metrics.score('/directory/of/images'))

# get ai corrupt score of list of images
print(metrics.score(['image1.png', 'image2.png']))

```

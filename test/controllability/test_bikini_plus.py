import os
import pathlib
import shutil

import pytest
from hbutils.system import TemporaryDirectory
from imgutils.data import load_image
from imgutils.sd import get_sdmeta_from_image

from sdeval.controllability import BikiniPlusMetrics
from test.testings import get_testfile


@pytest.fixture()
def bikini_plus_metrics():
    return BikiniPlusMetrics(
        tag_blacklist=[
            'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
            'breasts', 'grey_hair', 'medium_breasts'
        ]
    )


@pytest.fixture()
def bikini_plus_mldanbooru_metrics():
    return BikiniPlusMetrics(
        tagger='mldanbooru',
        tag_blacklist=[
            'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
            'breasts', 'grey_hair', 'medium_breasts'
        ]
    )


@pytest.fixture()
def bikini_plus_deepdanbooru_metrics():
    return BikiniPlusMetrics(
        tagger='deepdanbooru',
        tag_blacklist=[
            'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
            'breasts', 'grey_hair', 'medium_breasts'
        ]
    )


@pytest.fixture()
def bikini_image_files():
    return [
        get_testfile('bpscore', f'bikini-{i}.png')
        for i in range(15)
    ]


@pytest.fixture()
def bikini_image_dirs(bikini_image_files):
    with TemporaryDirectory() as td:
        dirs = []
        for i, file in enumerate(bikini_image_files):
            os.makedirs(os.path.join(td, str(i)), exist_ok=True)
            shutil.copyfile(file, os.path.join(td, str(i), 'bikini.png'))
            pathlib.Path(os.path.join(td, str(i), 'fuck.pt')).touch()
            dirs.append(os.path.join(td, str(i)))

        yield dirs


@pytest.fixture()
def bikini_image_prompts(bikini_image_files):
    retval = []
    for file in bikini_image_files:
        sdmeta = get_sdmeta_from_image(file)
        retval.append([(load_image(file), sdmeta.prompt, sdmeta.neg_prompt)])

    yield retval


@pytest.fixture()
def bikini_image_prompts_noneg(bikini_image_files):
    retval = []
    for file in bikini_image_files:
        sdmeta = get_sdmeta_from_image(file)
        retval.append((load_image(file), sdmeta.prompt))

    yield retval


@pytest.fixture()
def bikini_images(bikini_image_files):
    return [load_image(file) for file in bikini_image_files]


@pytest.mark.unittest
class TestControllabilityBikiniPlus:
    def test_score(self, bikini_plus_metrics, bikini_image_files):
        assert [bikini_plus_metrics.score(img_file) for img_file in bikini_image_files] == pytest.approx([
            0.8837757309353425, 0.8933908126091592, 0.9055491415894145, 0.8882521965374851, 0.8927720615148468,
            0.8469945459720423, 0.8399211360890133, 0.8098674415860692, 0.8363121274014674, 0.8389884182718645,
            0.8555319857366422, 0.8579074531926136, 0.8362479325036504, 0.839586421120691, 0.83640841923855,
        ])

    def test_score_dirs(self, bikini_image_dirs, bikini_plus_metrics):
        assert [bikini_plus_metrics.score(img_file) for img_file in bikini_image_dirs] == pytest.approx([
            0.8837757309353425, 0.8933908126091592, 0.9055491415894145, 0.8882521965374851, 0.8927720615148468,
            0.8469945459720423, 0.8399211360890133, 0.8098674415860692, 0.8363121274014674, 0.8389884182718645,
            0.8555319857366422, 0.8579074531926136, 0.8362479325036504, 0.839586421120691, 0.83640841923855,
        ])

    def test_score_prompts(self, bikini_image_prompts, bikini_plus_metrics):
        assert [bikini_plus_metrics.score(img_file) for img_file in bikini_image_prompts] == pytest.approx([
            0.8837757309353425, 0.8933908126091592, 0.9055491415894145, 0.8882521965374851, 0.8927720615148468,
            0.8469945459720423, 0.8399211360890133, 0.8098674415860692, 0.8363121274014674, 0.8389884182718645,
            0.8555319857366422, 0.8579074531926136, 0.8362479325036504, 0.839586421120691, 0.83640841923855,
        ])

    def test_score_prompts_noneg(self, bikini_image_prompts_noneg, bikini_plus_metrics):
        assert [bikini_plus_metrics.score(img_file) for img_file in bikini_image_prompts_noneg] == pytest.approx([
            0.8057833110347624, 0.8216747866489079, 0.8421623072638775, 0.8129909450109516, 0.8206196565969767,
            0.7439588227367002, 0.7331108945469114, 0.6815734980363065, 0.7266735161122132, 0.7308833770642613,
            0.7590933092927681, 0.7624429822550852, 0.7263612574844726, 0.731894281100077, 0.7342754910276161,
        ])

    def test_score_raw_images(self, bikini_images, bikini_plus_metrics):
        assert [bikini_plus_metrics.score(img_file) for img_file in bikini_images] == pytest.approx([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ])

    def test_score_mldanbooru(self, bikini_plus_mldanbooru_metrics, bikini_image_files):
        assert [bikini_plus_mldanbooru_metrics.score(img_file) for img_file in bikini_image_files] == pytest.approx([
            0.8702562209238103, 0.8719685555138217, 0.8684662004100342, 0.8753031082507153, 0.8647664087117215,
            0.8318993494506535, 0.8347541409924351, 0.8336778245243487, 0.8261908214946743, 0.8302784522734978,
            0.8242162834882449, 0.8223817627404567, 0.8299871676415369, 0.8199762032675165, 0.7998073322040836,
        ])

    def test_score_deepdanbooru(self, bikini_plus_deepdanbooru_metrics, bikini_image_files):
        assert [bikini_plus_deepdanbooru_metrics.score(img_file) for img_file in bikini_image_files] == pytest.approx([
            0.9186856502344132, 0.9099037922102063, 0.9178172856512113, 0.9052444168470831, 0.915857460324754,
            0.8565431211662344, 0.8414247717951927, 0.8604802844199334, 0.8504646470661714, 0.8524303560366839,
            0.8458638522556373, 0.841293096330979, 0.8226487314091085, 0.8368835960904908, 0.8290460122791237,
        ])

    def test_unknown_blacklist_tag(self):
        with pytest.warns(None):
            _ = BikiniPlusMetrics(
                tag_blacklist=[
                    'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
                    'breasts', 'grey_hair', 'medium_breasts',
                ]
            )

        with pytest.warns(UserWarning):
            _ = BikiniPlusMetrics(
                tag_blacklist=[
                    'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
                    'breasts', 'grey_hair', 'medium_breasts', ';sdjfljsdhfldsjgl;dfjglkdf',
                ]
            )

        with pytest.warns(UserWarning):
            _ = BikiniPlusMetrics(
                tag_blacklist=[
                    'bangs', 'long_hair', 'blue_eyes', 'animal_ears', 'sleeveless',
                    'breasts', 'grey_hair', 'medium_breasts', 'girl',
                ]
            )

    def test_failed(self, bikini_plus_metrics):
        with pytest.raises(FileNotFoundError):
            bikini_plus_metrics.score([])

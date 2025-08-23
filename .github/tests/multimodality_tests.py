import os

import pandas as pd
import pytest

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_LOCAL_TESTS = os.getenv("ENABLE_LOCAL_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "clip-ViT-B-32": ENABLE_LOCAL_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])

MODEL_NAME_TO_CLS = {
    "clip-ViT-B-32": SentenceTransformersRM,
    "gpt-4o-mini": LM,
}


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = MODEL_NAME_TO_CLS[model_path](model=model_path)

    return models


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        if not isinstance(model, LM):
            continue
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()


################################################################################
# Standard tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_filter_operation(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://thumbs.dreamstime.com/b/comida-r%C3%A1pida-nachos-con-el-sause-del-tomate-ejemplo-exhausto-de-la-acuarela-mano-aislado-en-blanco-150936354.jpg",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
    ]
    df = pd.DataFrame({"image": ImageArray(image_url)})
    user_instruction = "{image} represents food"
    filtered_df = df.sem_filter(user_instruction)

    expected_image_url = ImageArray(
        [
            "https://thumbs.dreamstime.com/b/comida-r%C3%A1pida-nachos-con-el-sause-del-tomate-ejemplo-exhausto-de-la-acuarela-mano-aislado-en-blanco-150936354.jpg",
        ]
    )

    assert len(filtered_df) < len(df)
    assert expected_image_url[0] in filtered_df["image"].values


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_join_operation(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
    ]
    elements = ["doll", "bird"]
    image_df = pd.DataFrame({"image": ImageArray(image_url)})
    element_df = pd.DataFrame({"element": elements})
    user_instruction = "{image} contains {element}"
    joined_df = image_df.sem_join(element_df, user_instruction)

    expected_result = [
        ("https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0", "doll"),
        ("https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg", "bird"),
    ]

    assert expected_result == list(zip(joined_df["image"], joined_df["element"]))


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_topk_operation(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://thumbs.dreamstime.com/b/comida-r%C3%A1pida-nachos-con-el-sause-del-tomate-ejemplo-exhausto-de-la-acuarela-mano-aislado-en-blanco-150936354.jpg",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
    ]
    df = pd.DataFrame({"image": ImageArray(image_url)})
    user_instruction = "{image} represents living beings"
    top_2_expected = set(
        [
            "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
            "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
        ]
    )

    methods = ["quick", "heap", "naive"]
    for method in methods:
        sorted_df = df.sem_topk(user_instruction, K=3, method=method)

        top_2_actual = set(sorted_df["image"].values)
        assert top_2_expected.issubset(top_2_actual)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_topk_with_groupby_operation(setup_models, model):
    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
    ]
    elements = ["doll", "bird"]
    image_df = pd.DataFrame({"image": ImageArray(image_url)})
    element_df = pd.DataFrame({"element": elements})

    df = image_df.join(element_df, how="cross")
    df.sem_topk("the {image} is most likely an {element}", K=1, group_by=["element"])
    assert len(set(df["element"])) == 2


@pytest.mark.parametrize("model", get_enabled("clip-ViT-B-32"))
def test_search_operation(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
    ]

    expected_result = set(["https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg"])

    df = pd.DataFrame({"image": ImageArray(image_url)})
    df = df.sem_index("image", "index_dir")
    df = df.sem_search("image", "bird", K=1)
    assert set(df["image"].values) == expected_result


@pytest.mark.parametrize("model", get_enabled("clip-ViT-B-32"))
def test_sim_join_operation_image_index(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://i1.wp.com/www.alloverthemap.net/wp-content/uploads/2014/02/2012-09-25-12.46.15.jpg?resize=400%2C284&amp;ssl=1",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
        "https://pravme.ru/wp-content/uploads/2018/01/sobor-Bogord-1.jpg",
    ]
    elements = ["doll", "bird"]

    image_df = pd.DataFrame({"image": ImageArray(image_url)}).sem_index("image", "index_dir")
    element_df = pd.DataFrame({"element": elements})

    joined_df = element_df.sem_sim_join(image_df, right_on="image", left_on="element", K=1)

    expected_result = [
        ("https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0", "doll"),
        ("https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg", "bird"),
    ]
    assert expected_result == list(zip(joined_df["image"], joined_df["element"]))


@pytest.mark.parametrize("model", get_enabled("clip-ViT-B-32"))
def test_sim_join_operation_text_index(setup_models, model):
    rm = setup_models[model]
    vs = FaissVS()
    lotus.settings.configure(rm=rm, vs=vs)

    image_url = [
        "https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0",
        "https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg",
    ]
    elements = ["doll", "bird"]

    image_df = pd.DataFrame({"image": ImageArray(image_url)})
    element_df = pd.DataFrame({"element": elements}).sem_index("element", "index_dir")

    joined_df = image_df.sem_sim_join(element_df, left_on="image", right_on="element", K=1)

    expected_result = [
        ("https://img.etsystatic.com/il/4bee20/1469037676/il_340x270.1469037676_iiti.jpg?version=0", "doll"),
        ("https://i.pinimg.com/236x/a4/3a/65/a43a65683a0314f29b66402cebdcf46d.jpg", "bird"),
    ]
    assert expected_result == list(zip(joined_df["image"], joined_df["element"]))

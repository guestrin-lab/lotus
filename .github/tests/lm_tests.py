import os

import pandas as pd
import pytest
from tokenizers import Tokenizer

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.types import CascadeArgs, PromptStrategy
from lotus.vector_store import FaissVS

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")

# Environment flags to enable/disable tests
ENABLE_OPENAI_TESTS = os.getenv("ENABLE_OPENAI_TESTS", "false").lower() == "true"
ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME_TO_ENABLED = {
    "gpt-4o-mini": ENABLE_OPENAI_TESTS,
    "gpt-4o": ENABLE_OPENAI_TESTS,
    "ollama/llama3.1": ENABLE_OLLAMA_TESTS,
}
ENABLED_MODEL_NAMES = set([model_name for model_name, is_enabled in MODEL_NAME_TO_ENABLED.items() if is_enabled])


def get_enabled(*candidate_models: str) -> list[str]:
    return [model for model in candidate_models if model in ENABLED_MODEL_NAMES]


@pytest.fixture(scope="session")
def setup_models():
    models = {}

    for model_path in ENABLED_MODEL_NAMES:
        models[model_path] = LM(model=model_path)

    return models


@pytest.fixture(autouse=True)
def print_usage_after_each_test(setup_models):
    yield  # this runs the test
    models = setup_models
    for model_name, model in models.items():
        print(f"\nUsage stats for {model_name} after test:")
        model.print_total_usage()
        model.reset_stats()
        model.reset_cache()


################################################################################
# Standard tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_filter_operation(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    data = {"Text": ["I am really excited to go to class today!", "I am very sad"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"
    filtered_df = df.sem_filter(user_instruction)

    expected_df = pd.DataFrame({"Text": ["I am really excited to go to class today!"]})
    assert filtered_df.equals(expected_df)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_topk(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Text": [
            "Lionel Messi is a good soccer player",
            "Michael Jordan is a good basketball player",
            "Steph Curry is a good basketball player",
            "Tom Brady is a good football player",
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "Which {Text} is most related to basketball?"
    top_2_expected = set(["Michael Jordan is a good basketball player", "Steph Curry is a good basketball player"])

    methods = ["quick", "heap", "naive"]
    for method in methods:
        sorted_df = df.sem_topk(user_instruction, K=2, method=method)

        top_2_actual = set(sorted_df["Text"].values)
        assert top_2_expected == top_2_actual


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_group_by_with_topk(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Course Name": [
            "Number Theory",
            "Data Structures",
            "Quantum Mechanics",
            "Genetics",
            "Linear Algebra",
            "Thermodynamics",
            "Algorithms",
            "Ecology",
            "Statistics",
            "Optics",
            "Machine Learning",
            "Molecular Biology",
        ],
        "Department": ["Math", "Physics", "Computer Science", "Biology"] * 3,
    }
    df = pd.DataFrame(data)
    user_instruction = "Which {Course Name} is the most challenging?"
    expected_df = pd.DataFrame(
        {
            "Course Name": ["Number Theory", "Thermodynamics", "Quantum Mechanics", "Molecular Biology"],
            "Department": ["Math", "Physics", "Computer Science", "Biology"],
        }
    )
    methods = ["quick", "heap", "naive"]
    for method in methods:
        sorted_df = df.sem_topk(user_instruction, K=1, method=method, group_by=["Department"])
        assert len(sorted_df) == 4
        assert set(sorted_df["Department"].values) == set(expected_df["Department"].values)
        assert set(sorted_df["Course Name"].values) == set(expected_df["Course Name"].values)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_join(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data1 = {"School": ["UC Berkeley", "Stanford"]}
    data2 = {"School Type": ["Public School", "Private School"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    join_instruction = "{School} is a {School Type}"
    joined_df = df1.sem_join(df2, join_instruction)
    joined_pairs = set(zip(joined_df["School"], joined_df["School Type"]))
    expected_pairs = set([("UC Berkeley", "Public School"), ("Stanford", "Private School")])
    assert joined_pairs == expected_pairs


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_map_fewshot(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {"School": ["UC Berkeley", "Carnegie Mellon"]}
    df = pd.DataFrame(data)
    examples = {"School": ["Stanford", "MIT"], "Answer": ["CA", "MA"]}
    examples_df = pd.DataFrame(examples)
    user_instruction = "What state is {School} in? Respond only with the two-letter abbreviation."
    df = df.sem_map(user_instruction, examples=examples_df, suffix="State")

    # clean up the state names to be more robust to free-form text
    df["State"] = df["State"].str[-2:].str.lower()
    pairs = set(zip(df["School"], df["State"]))
    expected_pairs = set([("UC Berkeley", "ca"), ("Carnegie Mellon", "pa")])
    assert pairs == expected_pairs


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_agg_then_map(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {"Text": ["My name is John", "My name is Jane", "My name is John"]}
    df = pd.DataFrame(data)
    agg_instruction = "What is the most common name in {Text}?"
    agg_df = df.sem_agg(agg_instruction, suffix="draft_output")
    assert len(agg_df) == 1

    map_instruction = "{draft_output} is a draft answer to the question 'What is the most common name?'. Clean up the draft answer so that there is just a single name. Your answer MUST be on word"
    cleaned_df = agg_df.sem_map(map_instruction, suffix="final_output")
    assert cleaned_df["final_output"].values[0].lower().strip(".,!?\"'") == "john"


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_group_by_with_agg(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Names": ["Michael", "Anakin", "Luke", "Dwight"],
        "Show": ["The Office", "Star Wars", "Star Wars", "The Office"],
    }
    df = pd.DataFrame(data)
    agg_instruction = "Summarize {Names}"
    agg_df: pd.DataFrame = df.sem_agg(agg_instruction, suffix="draft_output", group_by=["Show"])
    assert len(agg_df) == 2
    assert set(agg_df.columns.tolist()) == {"Show", "draft_output"}
    assert set(agg_df["Show"].values) == {"The Office", "Star Wars"}

    # Map post-processing
    map_instruction = "{draft_output} is a draft answer to the question 'Summarize the names'. Clean up the draft answer is just a comma separated list of names."
    cleaned_df = agg_df.sem_map(map_instruction, suffix="final_output")

    assert set(cleaned_df["final_output"].values[0].lower().strip(".,!?\"'").split(", ")) == {"anakin", "luke"}
    assert set(cleaned_df["final_output"].values[1].lower().strip(".,!?\"'").split(", ")) == {"michael", "dwight"}


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_sem_extract(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    data = {
        "Text": [
            "Lionel Messi is a good soccer player, he has won the World Cup 5 times",
            "Michael Jordan is a good basketball player, he has won the NBA championships 6 times",
            "Tiger Woods is a good golf player, he has won the Master championships 4 times",
            "Tom Brady is a good football player, he has won the NFL championships 7 times",
        ]
    }
    df = pd.DataFrame(data)
    input_cols = ["Text"]
    output_cols = {
        "Name": None,
        "Sport": None,
        "Number of Championships": None,
    }
    df = df.sem_extract(input_cols, output_cols, extract_quotes=True)

    expected_values = {
        "Name": ["lionel messi", "michael jordan", "tiger woods", "tom brady"],
        "Sport": ["soccer", "basketball", "golf", "football"],
        "Number of Championships": ["5", "6", "4", "7"],
    }

    for col in output_cols:
        assert [str(val).strip().lower() for val in df[col].tolist()] == expected_values[col]

    for idx, row in df.iterrows():
        assert row["Name"] in row["Name_quote"], f"Name '{row['Name']}' not found in '{row['Name_quote']}'"
        assert (
            row["Sport"].lower() in row["Sport_quote"].lower()
        ), f"Sport '{row['Sport']}' not found in '{row['Sport_quote']}'"
        assert (
            str(row["Number of Championships"]) in row["Number of Championships_quote"]
        ), f"Number of Championships '{row['Number of Championships']}' not found in '{row['Number of Championships_quote']}'"


################################################################################
# CoT tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_filter_operation_cot(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    data = {
        "Text": [
            "I had two apples, then I gave away one",
            "My friend gave me an apple",
            "I gave away both of my apples",
            "I gave away my apple, then a friend gave me his apple, then I threw my apple away",
        ]
    }
    df = pd.DataFrame(data)
    user_instruction = "{Text} I have at least one apple"
    filtered_df = df.sem_filter(user_instruction, prompt_strategy=PromptStrategy(cot=True))
    expected_df = pd.DataFrame({"Text": ["I had two apples, then I gave away one", "My friend gave me an apple"]})
    assert filtered_df.equals(expected_df)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_filter_operation_cot_fewshot(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    data = {
        "Sequence": [
            "Five, Four, Three",
            "A, B, C",
            "Pond, Lake, Ocean",
        ]
    }
    df = pd.DataFrame(data)
    examples = {
        "Sequence": ["1, 2, 3", "A, B, C", "penny, nickel, dime, quarter", "villiage, town, city"],
        "Answer": [True, True, True, True],
        "Reasoning": [
            "1, 2, 3 is an increasing sequence of numbers",
            "A, B, C is an increasing sequence of letters in alphabetical order",
            "penny, nickel, dime, quarter is an increasing sequence of coins by value",
            "villiage, town, city is an increasing sequence of settlements",
        ],
    }
    examples_df = pd.DataFrame(examples)

    user_instruction = "{Sequence} is increasing"
    filtered_df = df.sem_filter(
        user_instruction,
        prompt_strategy=PromptStrategy(cot=True),
        examples=examples_df,
        additional_cot_instructions="Assume the most typical or logical case.",
    )
    expected_df = pd.DataFrame(
        {
            "Sequence": [
                "A, B, C",
                "Pond, Lake, Ocean",
            ]
        },
        index=[1, 2],
    )
    assert filtered_df.equals(expected_df)


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_filter_operation_cot_fewshot_no_reasoning(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    # Test filter operation on an easy dataframe
    data = {
        "Sequence": [
            "Five, Four, Three",
            "A, B, C",
            "Pond, Lake, Ocean",
        ]
    }
    df = pd.DataFrame(data)
    examples = {
        "Sequence": ["1, 2, 3", "penny, nickel, dime, quarter", "villiage, town, city", "A, B, C"],
        "Answer": [True, True, True, True],
    }
    examples_df = pd.DataFrame(examples)

    user_instruction = "{Sequence} is increasing"
    filtered_df = df.sem_filter(user_instruction, prompt_strategy=PromptStrategy(cot=True), examples=examples_df)
    expected_df = pd.DataFrame(
        {
            "Sequence": [
                "A, B, C",
                "Pond, Lake, Ocean",
            ]
        },
        index=[1, 2],
    )
    assert filtered_df.equals(expected_df)


################################################################################
# Cascade tests
################################################################################
@pytest.mark.skipif(not ENABLE_OPENAI_TESTS, reason="Skipping test because OpenAI tests are not enabled")
def test_filter_cascade(setup_models):
    models = setup_models
    lotus.settings.configure(lm=models["gpt-4o"], helper_lm=models["gpt-4o-mini"])

    data = {
        "Text": [
            # Positive examples
            "I am really excited to go to class today!",
            "Today is going to be an amazing day!",
            "I absolutely love the new project I am working on.",
            "Feeling so grateful for everything I have.",
            "I can't wait to see my friends this weekend!",
            "The weather is beautiful, and I feel fantastic.",
            "Just received some great news about my promotion!",
            "I'm so happy to have such supportive colleagues.",
            "I'm thrilled to be learning something new every day.",
            "Life is really good right now, and I feel blessed.",
            "I am proud of all the progress I've made this year.",
            "Today was productive, and I feel accomplished.",
            "I’m really enjoying my workout routine lately!",
            "Got a compliment from my manager today, feeling awesome!",
            "Looking forward to spending time with family tonight.",
            "Just finished a great book and feel inspired!",
            "Had a lovely meal with friends, life is good!",
            "Everything is going as planned, couldn't be happier.",
            "Feeling super motivated and ready to take on challenges!",
            "I appreciate all the small things that bring me joy.",
            # Negative examples
            "I am very sad.",
            "Today has been really tough; I feel exhausted.",
            "I'm feeling pretty down about how things are going.",
            "I’m overwhelmed with all these challenges.",
            "It’s hard to stay positive when things keep going wrong.",
            "I feel so alone and unappreciated.",
            "My energy is low, and nothing seems to cheer me up.",
            "Feeling anxious about everything lately.",
            "I’m disappointed with the way my project turned out.",
            "Today has been one of those days where everything goes wrong.",
            "Life feels really overwhelming right now.",
            "I can't seem to find any motivation these days.",
            "I’m worried about the future and what it holds.",
            "It's been a stressful day, and I feel mentally drained.",
            "I feel like I'm falling behind everyone else.",
            "Just can't seem to catch a break recently.",
            "I’m really struggling to keep up with all my responsibilities.",
            "Had an argument with a close friend, feeling hurt.",
            "I don’t feel supported by my team at work.",
            "Life has been tough lately, and I’m feeling down.",
        ]
    }

    df = pd.DataFrame(data)
    user_instruction = "{Text} is a positive sentiment"

    # All filters resolved by the helper model
    filtered_df, stats = df.sem_filter(
        user_instruction=user_instruction,
        cascade_args=CascadeArgs(
            learn_cascade_threshold_sample_percentage=0.5,
            recall_target=0.9,
            precision_target=0.9,
            failure_probability=0.2,
        ),
        return_stats=True,
    )

    assert "I am really excited to go to class today!" in filtered_df["Text"].values
    assert "I am very sad" not in filtered_df["Text"].values
    assert stats["filters_resolved_by_helper_model"] > 0, stats


@pytest.mark.skipif(not ENABLE_OPENAI_TESTS, reason="Skipping test because OpenAI tests are not enabled")
def test_join_cascade(setup_models):
    models = setup_models
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    vs = FaissVS()
    lotus.settings.configure(lm=models["gpt-4o-mini"], rm=rm, vs=vs)

    data1 = {
        "School": [
            "University of California, Berkeley",
            "Stanford University",
            "Carnegie Mellon University",
            "Massachusetts Institute of Technology (MIT)",
            "Harvard University",
            "University of Michigan",
            "California Institute of Technology (Caltech)",
            "University of Illinois Urbana-Champaign",
            "Princeton University",
            "University of Texas at Austin",
            "University of Chicago",
            "University of Washington",
            "Yale University",
            "Cornell University",
            "University of Pennsylvania",
        ]
    }
    data2 = {"School Type": ["Public School", "Private School"]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    join_instruction = "{School} is a {School Type}"
    expected_pairs = [
        ("University of California, Berkeley", "Public School"),
        ("Stanford University", "Private School"),
    ]

    # Cascade join
    joined_df, stats = df1.sem_join(
        df2,
        join_instruction,
        cascade_args=CascadeArgs(
            recall_target=0.7, precision_target=0.7, min_join_cascade_size=10, cascade_IS_random_seed=42
        ),
        return_stats=True,
    )

    for pair in expected_pairs:
        school, school_type = pair
        exists = ((joined_df["School"] == school) & (joined_df["School Type"] == school_type)).any()
        assert exists, f"Expected pair {pair} does not exist in the dataframe!"
    assert stats["join_resolved_by_helper_model"] > 0, stats

    # All joins resolved by the large model
    joined_df, stats = df1.sem_join(
        df2,
        join_instruction,
        cascade_args=CascadeArgs(
            recall_target=1.0, precision_target=1.0, min_join_cascade_size=10, cascade_IS_random_seed=42
        ),
        return_stats=True,
    )

    for pair in expected_pairs:
        school, school_type = pair
        exists = ((joined_df["School"] == school) & (joined_df["School Type"] == school_type)).any()
        assert exists, f"Expected pair {pair} does not exist in the dataframe!"
    assert (
        stats["join_resolved_by_large_model"] > stats["join_resolved_by_helper_model"]
    ), stats  # helper negative still can still meet the precision target
    assert stats["join_helper_positive"] == 0, stats


@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini"))
def test_format_logprobs_for_filter_cascade(setup_models, model):
    lm = setup_models[model]
    messages = [
        [{"role": "user", "content": "True or False: The sky is blue?"}],
    ]
    response = lm(messages, logprobs=True)
    formatted_logprobs = lm.format_logprobs_for_filter_cascade(response.logprobs)
    true_probs = formatted_logprobs.true_probs
    assert len(true_probs) == 1

    # Very safe (in practice its ~1)
    assert true_probs[0] > 0.8
    assert len(formatted_logprobs.tokens) == len(formatted_logprobs.confidences)


################################################################################
# Token counting tests
################################################################################
@pytest.mark.parametrize("model", get_enabled("gpt-4o-mini", "ollama/llama3.1"))
def test_count_tokens(setup_models, model):
    lm = setup_models[model]
    lotus.settings.configure(lm=lm)

    tokens = lm.count_tokens("Hello, world!")
    assert lm.count_tokens([{"role": "user", "content": "Hello, world!"}]) == tokens
    assert tokens < 100


def test_custom_tokenizer():
    custom_tokenizer = Tokenizer.from_pretrained("gpt2")
    custom_lm = LM(model="doesn't matter", tokenizer=custom_tokenizer)
    tokens = custom_lm.count_tokens("Hello, world!")
    assert custom_lm.count_tokens([{"role": "user", "content": "Hello, world!"}]) == tokens
    assert tokens < 100

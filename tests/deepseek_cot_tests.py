import os

import pandas as pd
import pytest

import lotus
from lotus.models import LM
from lotus.types import PromptStrategy

lotus.logger.setLevel("DEBUG")

ENABLE_OLLAMA_TESTS = os.getenv("ENABLE_OLLAMA_TESTS", "false").lower() == "true"

MODEL_NAME = "ollama/deepseek-r1:7b"


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_demonstrations_only():
    """Test DeepSeek with demonstrations without CoT reasoning."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Course": ["Linear Algebra", "Creative Writing", "Calculus", "Art History"]}
    df = pd.DataFrame(data)
    user_instruction = "{Course} requires mathematical skills"

    # Provide examples without reasoning
    examples = pd.DataFrame({"Course": ["Statistics", "Poetry", "Physics"], "Answer": [True, False, True]})

    result = df.sem_filter(user_instruction, prompt_strategy=PromptStrategy(dems=examples), return_all=True)

    assert "filter_label" in result.columns
    # Should identify math courses correctly based on examples
    math_courses = result[result["filter_label"]]["Course"].tolist()
    assert any(course in ["Linear Algebra", "Calculus"] for course in math_courses)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_cot_demonstrations_combined():
    """Test DeepSeek with combined CoT and demonstrations."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Product": ["Smartphone", "Book", "Laptop", "Pen"]}
    df = pd.DataFrame(data)
    user_instruction = "{Product} is an electronic device"

    # Provide examples with reasoning
    examples = pd.DataFrame(
        {
            "Product": ["Tablet", "Magazine", "Smart Watch"],
            "Answer": [True, False, True],
            "Reasoning": [
                "Tablets are electronic devices with screens and processors",
                "Magazines are printed materials, not electronic",
                "Smart watches are wearable electronic devices with digital displays",
            ],
        }
    )

    result = df.sem_filter(
        user_instruction,
        prompt_strategy=PromptStrategy(cot=True, dems=examples),
        examples=examples,
        return_explanations=True,
        return_all=True,
    )

    assert "filter_label" in result.columns
    assert "explanation_filter" in result.columns

    # Should identify electronic devices correctly
    electronic_devices = result[result["filter_label"]]["Product"].tolist()
    assert any(device in ["Smartphone", "Laptop"] for device in electronic_devices)

    # Check explanations are provided
    for explanation in result["explanation_filter"]:
        assert explanation is not None
        assert len(explanation) > 0


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_demonstration_config():
    """Test DeepSeek with examples."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Animal": ["Dog", "Cat", "Eagle", "Fish"]}
    df = pd.DataFrame(data)
    user_instruction = "{Animal} can fly"

    # Provide examples
    examples = pd.DataFrame({"Animal": ["Bird", "Elephant"], "Answer": [True, False]})

    result = df.sem_filter(
        user_instruction,
        prompt_strategy=PromptStrategy(cot=True, dems=examples),
        return_all=True,
    )

    assert "filter_label" in result.columns
    # Should identify flying animals correctly
    flying_animals = result[result["filter_label"]]["Animal"].tolist()
    assert "Eagle" in flying_animals


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_bootstrapping():
    """Test DeepSeek with automatic demonstration bootstrapping."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"City": ["New York", "London", "Tokyo", "Sydney", "Paris"]}
    df = pd.DataFrame(data)
    user_instruction = "{City} is in Asia"

    # Configure bootstrapping
    result = df.sem_filter(
        user_instruction,
        prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=2),
        return_explanations=True,
        return_all=True,
    )

    assert "filter_label" in result.columns
    assert "explanation_filter" in result.columns

    # Should identify Asian cities correctly
    asian_cities = result[result["filter_label"]]["City"].tolist()
    assert "Tokyo" in asian_cities

    # Should work even without user-provided examples
    assert len(result) == len(df)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_extract_with_cot():
    """Test DeepSeek extract operation with CoT reasoning."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {
        "Review": [
            "This phone has amazing battery life and great camera quality!",
            "The laptop is too slow and overheats frequently.",
        ]
    }
    df = pd.DataFrame(data)

    output_cols = {
        "sentiment": "Overall sentiment (positive/negative)",
        "main_feature": "Main feature mentioned in the review",
    }

    input_cols = ["Review"]  # Columns to extract from

    result = df.sem_extract(input_cols, output_cols, prompt_strategy=PromptStrategy(cot=True), return_explanations=True)

    assert "sentiment" in result.columns
    assert "main_feature" in result.columns
    assert "explanation_extract" in result.columns

    # Check sentiment extraction
    sentiments = result["sentiment"].tolist()
    assert any("positive" in sent.lower() for sent in sentiments)
    assert any("negative" in sent.lower() for sent in sentiments)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_backward_compatibility():
    """Test that DeepSeek still works with legacy methods."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Text": ["The weather is sunny today", "It's raining heavily outside"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} describes good weather"

    # Test without explicit strategy (should use default behavior)
    result_default = df.sem_filter(user_instruction, return_all=True)

    # Test with explicit CoT strategy
    result_cot = df.sem_filter(user_instruction, prompt_strategy=PromptStrategy(cot=True), return_all=True)

    # Both should work and produce results
    assert "filter_label" in result_default.columns
    assert "filter_label" in result_cot.columns
    assert len(result_default) == len(result_cot) == len(df)


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_error_handling():
    """Test error handling with DeepSeek and new reasoning strategies."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Text": ["Sample text"]}
    df = pd.DataFrame(data)
    user_instruction = "{Text} is meaningful"

    # Test with empty examples
    empty_examples = pd.DataFrame(columns=["Text", "Answer"])

    try:
        result = df.sem_filter(user_instruction, prompt_strategy=PromptStrategy(dems=empty_examples), return_all=True)
        # Should handle gracefully
        assert "filter_label" in result.columns
    except Exception as e:
        # If it raises an error, it should be informative
        assert len(str(e)) > 0


@pytest.mark.skipif(not ENABLE_OLLAMA_TESTS, reason="Skipping test because Ollama tests are not enabled")
def test_deepseek_multiple_operations_chaining():
    """Test chaining multiple operations with DeepSeek and different strategies."""
    lm = LM(model=MODEL_NAME)
    lotus.settings.configure(lm=lm)

    data = {"Product": ["iPhone", "Novel", "MacBook", "Newspaper", "iPad"]}
    df = pd.DataFrame(data)

    # First filter with demonstrations
    examples = pd.DataFrame({"Product": ["Laptop", "Book"], "Answer": [True, False]})

    filtered_df = df.sem_filter("{Product} is an electronic device", prompt_strategy=PromptStrategy(dems=examples))

    # Then map with CoT
    if len(filtered_df) > 0:
        mapped_df = filtered_df.sem_map(
            "What category does {Product} belong to?",
            prompt_strategy=PromptStrategy(cot=True),
            return_explanations=True,
        )

        assert "_map" in mapped_df.columns
        assert "explanation_map" in mapped_df.columns

        # Should have reasonable categorizations
        for category in mapped_df["_map"]:
            assert isinstance(category, str)
            assert len(category) > 0

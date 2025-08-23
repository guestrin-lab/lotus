import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import PromptStrategy

# Configure the language model
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Sample data
data = {
    "Course Name": ["Linear Algebra", "Poetry Writing", "Calculus II", "Art History", "Statistics", "Creative Writing"]
}
df = pd.DataFrame(data)
user_instruction = "{Course Name} requires a lot of math"

# Example 1: Basic filtering (no reasoning)
print("=== 1. Basic Filtering ===")
basic_df = df.sem_filter(user_instruction, return_all=True)
print(basic_df[["Course Name", "filter_label"]])
print()

# Example 2: Chain-of-Thought reasoning
print("=== 2. Chain-of-Thought Reasoning ===")
cot_df = df.sem_filter(
    user_instruction, prompt_strategy=PromptStrategy(cot=True), return_explanations=True, return_all=True
)
print(cot_df[["Course Name", "filter_label", "explanation_filter"]])
print()

# Example 3: Few-shot examples (demonstrations)
print("=== 3. Few-shot Examples ===")
examples = pd.DataFrame({"Course Name": ["Machine Learning", "Literature", "Physics"], "Answer": [True, False, True]})

demo_df = df.sem_filter(
    user_instruction,
    prompt_strategy=PromptStrategy(dems=examples),
    return_all=True,
)
print(demo_df[["Course Name", "filter_label"]])
print()

# Example 4: CoT + Demonstrations (the powerful combination)
print("=== 4. CoT + Demonstrations ===")
examples_with_reasoning = pd.DataFrame(
    {
        "Course Name": ["Machine Learning", "Literature", "Physics"],
        "Answer": [True, False, True],
        "Reasoning": [
            "Machine Learning requires linear algebra, calculus, and statistics",
            "Literature focuses on reading, writing, and analysis - no math required",
            "Physics is fundamentally mathematical with equations and calculations",
        ],
    }
)

combined_df = df.sem_filter(
    user_instruction,
    prompt_strategy=PromptStrategy(cot=True, dems=examples_with_reasoning),
    return_explanations=True,
    return_all=True,
)
print(combined_df[["Course Name", "filter_label", "explanation_filter"]])
print()

# Example 5: Automatic demonstration bootstrapping
print("=== 5. Bootstrapped Demonstrations ===")

bootstrap_df = df.sem_filter(
    user_instruction,
    prompt_strategy=PromptStrategy(cot=True, dems="auto", max_dems=2),
    return_explanations=True,
    return_all=True,
)
print("Automatically generated demonstrations:")
print(bootstrap_df[["Course Name", "filter_label", "explanation_filter"]])
print()

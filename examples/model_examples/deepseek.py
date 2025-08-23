import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import PromptStrategy

# Set up model
lm = LM(model="ollama/deepseek-r1:7b", temperature=0.6)
lotus.settings.configure(lm=lm)

# Input data: product reviews
data = {
    "Product Name": [
        "Sony WH-1000XM5",
        "Bose QuietComfort 45",
        "Apple AirPods Max",
        "Sennheiser Momentum 4 Wireless",
    ],
    "Review": [
        "Excellent noise cancellation and very comfortable. Battery life is great for travel.",
        "Superb sound quality and lightweight design. Pairs quickly and feels premium.",
        "Rich spatial audio experience, but a bit heavier. Strong integration with iOS ecosystem.",
        "Clean, balanced sound profile with long battery life. Slightly bulky but well-built.",
    ],
}

df = pd.DataFrame(data)

# Instruction: Find a similar product based on the review
user_instruction = (
    "Given the review for {Product Name}, identify which other product in the list is most similar in terms of "
    "sound quality and user experience. Just return the most similar product name."
)

# Run semantic mapping with CoT strategy
df = df.sem_map(user_instruction, return_explanations=True, prompt_strategy=PromptStrategy(cot=True))

print(df)

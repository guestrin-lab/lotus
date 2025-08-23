import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import PromptStrategy

lm = LM(model="gpt-4o-mini")

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
# Old way: filtered_df = df.sem_filter(user_instruction, strategy="cot", return_all=True)
filtered_df = df.sem_filter(
    user_instruction, prompt_strategy=PromptStrategy(cot=True), return_all=True, return_explanations=True
)  # uncomment to see reasoning chains

print(filtered_df)
# print(filtered_df)

import pandas as pd

import lotus
from lotus.models import LM

lm = LM(model="ollama/deepseek-r1:7b")
lotus.settings.configure(lm=lm)

df = pd.DataFrame(
    {
        "description": [
            "Yoshi is 25 years old",
            "Bowser is 45 years old",
            "Luigi is 15 years old",
        ]
    }
)
input_cols = ["description"]

# A description can be specified for each output column
output_cols = {
    "masked_col_1": "The name of the person",
    "masked_col_2": "The age of the person",
}

# you can optionally set extract_quotes=True to return quotes that support each output
new_df = df.sem_extract(input_cols, output_cols, extract_quotes=True, return_explanations=True)
print(new_df)

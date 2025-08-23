import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import PromptStrategy

lm = LM(model="ollama/deepseek-r1:7b", temperature=0.6)
lotus.settings.configure(lm=lm)

data = {
    "Review": [
        "This vacuum cleaner is the best I've ever owned. Highly recommend it!",
        "It's okay, not sure I would buy it again.",
        "Terrible experience, broke after a few uses.",
        "Amazing build quality and customer support. Would absolutely recommend.",
        "I would not recommend this to anyone.",
    ]
}

df = pd.DataFrame(data)

for method in ["quick", "heap", "naive"]:
    sorted_df, stats = df.sem_topk(
        "{Review} suggests that the user would recommend the product to others",
        K=2,
        method=method,
        prompt_strategy=PromptStrategy(cot=True),
        return_stats=True,
        return_explanations=True,
    )
    print(sorted_df)
    print(stats)

import pandas as pd

import lotus
from lotus.models import LM
from lotus.sem_ops.postprocessors import deepseek_cot_postprocessor

lm = LM(model="ollama/deepseek-r1:7b")

lotus.settings.configure(lm=lm)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
    ]
}
df = pd.DataFrame(data)
user_instruction = "What is a similar course to {Course Name}. Just give the course name."
df = df.sem_map(user_instruction, return_explanations=True, reasoning_parser=deepseek_cot_postprocessor)
print(df)

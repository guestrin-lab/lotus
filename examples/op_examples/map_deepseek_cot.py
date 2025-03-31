import pandas as pd

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

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
df = df.sem_map(user_instruction, return_explanations=True, strategy=ReasoningStrategy.ZS_COT)
print(df)

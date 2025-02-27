import pandas as pd

import lotus
from lotus.models import LM

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
df = df.sem_map(user_instruction, strategy="zs-cot", return_explanations=True, model_type="deepseek")
print(df)

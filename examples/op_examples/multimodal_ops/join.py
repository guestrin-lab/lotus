import os

import pandas as pd

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.models import LM
from lotus.types import PromptStrategy

lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

# The images folder contain images representing digits taken from MNIST dataset
image_file_names = os.listdir("images")  # get all file in the folder

# file names are the same as the digit represented by image
image_paths = [os.path.join("images", image) for image in image_file_names]

image_df = pd.DataFrame({"image": ImageArray(image_paths), "image_path": image_paths})
labels_df = pd.DataFrame({"label": [0, 1]})

df = image_df.sem_join(labels_df, "{image} represents the number {label}", prompt_strategy=PromptStrategy(cot=True))

print(df)

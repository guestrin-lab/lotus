# Contributing

## Setting up

To set up for development, create a conda environment, install lotus, and install additional dev dependencies.
```
conda env create -f environment.yml
conda activate lotus
git clone git@github.com:stanford-futuredata/lotus.git
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

## Dev Flow
After making your changes, please make a PR to get your changes merged upstream.

## Running Models
To run a model, you can use the `LM` class in `lotus.models.LM`. We use the `litellm` library to interface with the model.
This allows you to use any model provider that is supported by `litellm`.

Here's an example of creating an `LM` object for `gpt-4o`
```
from lotus.models import LM
lm = LM(model="gpt-4o")
```

Here's an example of creating an `LM` object to use `llama3.2` on Ollama
```
from lotus.models import LM
lm = LM(model="ollama/llama3.2")
```

Here's an example of creating an `LM` object to use `Meta-Llama-3-8B-Instruct` on vLLM
```
from lotus.models import LM
lm = LM(model='hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct',
        api_base='http://localhost:8000/v1',
        max_ctx_len=8000,
        max_tokens=1000)
```

## Helpful Examples
For helpful examples of LOTUS operators, please refer to the `examples` folder, as well as the documentation.

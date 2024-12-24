# LOTUS:  A Query Engine For Processing Data with LLMs
<!--- BADGES: START --->
<!--[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OzoJXH13aOwNOIEemClxzNCNYnqSGxVl?usp=sharing)-->
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qtmklXD_J1SSJLR86ws4GsYcLln6FZqY?usp=sharing#scrollTo=p5YByUTZqUqN)
[![Arxiv](https://img.shields.io/badge/arXiv-2407.11418-B31B1B.svg)][#arxiv-paper-package]
[![Slack](https://img.shields.io/badge/slack-lotus-purple.svg?logo=slack)][#slack]
[![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)](https://lotus-ai.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)][#pypi-package]
[![PyPI](https://img.shields.io/pypi/v/lotus-ai)][#pypi-package]
[![GitHub license](https://img.shields.io/badge/License-MIT-blu.svg)][#license-gh-package]

[#license-gh-package]: https://lbesson.mit-license.org/
[#arxiv-paper-package]: https://arxiv.org/abs/2407.11418
[#pypi-package]: https://pypi.org/project/lotus-ai/
[#slack]: https://join.slack.com/t/lotus-fnm8919/shared_invite/zt-2tnq6948j-juGuSIR0__fsh~kUmZ6TJw
<!--- BADGES: END --->

LOTUS makes LLM-powered data processing fast and easy. 

LOTUS (**L**LMs **O**ver **T**ables of **U**nstructured and **S**tructured Data) provides a declarative programming model and an optimized query engine for serving powerful reasoning-based query pipelines over structured and unstructured data! We provide a simple and intuitive Pandas-like API, that implements **semantic operators**. 

For trouble-shooting or feature requests, please raise an issue and we'll get to it promptly. To share feedback and applications you're working on, you can send us a message on our [community slack](https://join.slack.com/t/lotus-fnm8919/shared_invite/zt-2tnq6948j-juGuSIR0__fsh~kUmZ6TJw), or send an email (lianapat@stanford.edu).

# Installation
```
conda env create -f environment.yml
conda activate lotus
pip install lotus-ai
```

## Running on Mac
If you are running on mac, please install Faiss via conda:

### CPU-only version
```
conda install -c pytorch faiss-cpu=1.8.0
```

### GPU(+CPU) version
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
For more details, see [Installing FAISS via Conda](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda).

# Quickstart
If you're already familiar with Pandas, getting started will be a breeze! Below we provide a simple example program using the semantic join operator. The join, like many semantic operators, are specified by **langex** (natural language expressions), which the programmer uses to specify the operation. Each langex is parameterized by one or more table columns, denoted in brackets. The join's langex serves as a predicate and is parameterized by a right and left join key.
```python
import pandas as pd
import lotus
from lotus.models import LM

# configure the LM, and remember to export your API key
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# create dataframes with course names and skills
courses_data = {
    "Course Name": [
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}
skills_data = {"Skill": ["Math", "Computer Science"]}
courses_df = pd.DataFrame(courses_data)
skills_df = pd.DataFrame(skills_data)

# lotus sem join 
res = courses_df.sem_join(skills_df, "Taking {Course Name} will help me learn {Skill}")
print(res)

# Print total LM usage
lm.print_total_usage()
```


## Key Concept: The Semantic Operator Model
LOTUS' implements is the semantic operator programming model. Semantic operators as declarative transformations on one or more datasets, parameterized by a natural language expression, that can be implemented by a variety of AI-based algorithms. Semantic operators seamlessly extend the relational model, operating over tables that may contain traditional structured data as well as unstructured fields, such as free-form text. These composable, modular language- based operators allow you to write AI-based pipelines with high-level logic, leaving the rest of the work to the query engine! Each operator can be implemented and optimized in multiple ways, opening a rich space for execution plans, similar to relational operators. To learn more about the semantic operator model, read the full [research paper](https://arxiv.org/abs/2407.11418).

LOTUS offers a number of semantic operators in a Pandas-like API, some of which are described below. To learn more about semantic operators provided in LOTUS, check out the full [documentation](https://lotus-ai.readthedocs.io/en/latest/), run the [colab tutorial](https://colab.research.google.com/drive/1OzoJXH13aOwNOIEemClxzNCNYnqSGxVl?usp=sharing), or you can also refer to these [examples](https://github.com/TAG-Research/lotus/tree/main/examples/op_examples).

| Operator   | Description                                     |
|------------|-------------------------------------------------|
| sem_map      |  Map each record using a natural language projection| 
| sem_filter   | Keep records that match the natural language predicate |  
| sem_extract  | Extract one or more attributes from each row        |
| sem_agg      | Aggregate across all records (e.g. for summarization)             |
| sem_topk     | Order the records by some natural langauge sorting criteria                 |
| sem_join     | Join two datasets based on a natural language predicate       |
| sem_sim_join | Join two DataFrames based on semantic similarity             |
| sem_search   | Perform semantic search the over a text column                |


# Supported Models
There are 3 main model classes in LOTUS:
- `LM`: The language model class.
    - The `LM` class is built on top of the `LiteLLM` library, and supports any model that is supported by `LiteLLM`. See [this page](CONTRIBUTING.md) for examples of using models on `OpenAI`, `Ollama`, and `vLLM`. Any provider supported by `LiteLLM` should work. Check out [litellm's documentation](https://litellm.vercel.app) for more information.
- `RM`: The retrieval model class.
    - Any model from `SentenceTransformers` can be used with the `SentenceTransformersRM` class, by passing the model name to the `model` parameter (see [an example here](examples/op_examples/dedup.py)). Additionally, `LiteLLMRM` can be used with any model supported by `LiteLLM` (see [an example here](examples/op_examples/sim_join.py)).
- `Reranker`: The reranker model class.
    - Any `CrossEncoder` from `SentenceTransformers` can be used with the `CrossEncoderReranker` class, by passing the model name to the `model` parameter (see [an example here](examples/op_examples/search.py)).

# Feature Requests and Contributing
If you have a feature request, we're happy to hear from you! Please open an issue.

If you're interested in contributing, we'd be happy to coordinate on ongoing efforts! Please send an email to Liana (lianapat@stanford.edu) or reach out on our [slack](https://join.slack.com/t/lotus-fnm8919/shared_invite/zt-2tnq6948j-juGuSIR0__fsh~kUmZ6TJw). 

# References
For recent updates related to LOTUS, follow [@lianapatel_](https://x.com/lianapatel_) on X.

If you find LOTUS or semantic operators useful, we'd appreciate if you can please cite this work as follows:
```bibtex
@misc{patel2024semanticoperators,
      title={Semantic Operators: A Declarative Model for Rich, AI-based Analytics Over Text Data},
      author={Liana Patel and Siddharth Jha and Parth Asawa and Melissa Pan and Carlos Guestrin and Matei Zaharia},
      year={2024},
      eprint={2407.11418},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2407.11418},
}
```

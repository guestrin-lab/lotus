import os

import pandas as pd

import lotus
from lotus import WebSearchCorpus, parse_pdf, web_search
from lotus.models import LM

os.environ["SERPAPI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

lm = LM(model="gpt-4o-mini")

lotus.settings.configure(lm=lm)

"""
Creating a deep research pipeline with LOTUS is as simple as two commands -- `web_search` and `sem_agg`. 
We can use `web_search` to pull a list of relevant articles from Google. From here, we can use `sem_agg` to 
condense the articles into a more concise response. This directly makes use of Internet sources to make the 
most accurate and up-to-date response. 
"""

question = "Why do attention-based models like Transformers scale better than CNNs or RNNs for large datasets?"
# Pull the first five relevant articles from Google
df_google = web_search(WebSearchCorpus.GOOGLE, question, 5)[["title", "snippet"]]
print(f"Results from Google\n{df_google}")

# Condense or summarize the articles
df_agg_result = df_google.sem_agg(
    "Summarize each {snippet} into a brief paragraph, answering the question: " + question
)
print(f"Summarized results\n{df_agg_result._output[0]}")

"""
We can also use `web_search` with arXiv to pull a list of relevant research papers for more accurate research.
Let's try to answer the same question once again, but using both Google and arXiv.
"""

df_arxiv = web_search(WebSearchCorpus.ARXIV, question, 5)[
    ["title", "abstract"]
]  # Working with arXiv, we parse the `abstract` of a paper instead of `snippet`.
print(f"Results from arXiv\n{df_arxiv}")

df_arxiv = df_arxiv.rename(columns={"abstract": "snippet"})
df_altogether = pd.concat([df_google, df_arxiv])
df_joint_agg_result = df_altogether.sem_agg(
    "Summarize each {snippet} into a brief paragraph, answering the question: " + question
)
print(f"Summarized results\n{df_joint_agg_result._output[0]}")

"""
Sometimes, less is more. Not all these sources will be equally relevant to the question. Now, let's expand to more than just `web_search` and `sem_agg` 
to see if we can filter by relevance. Let's only take the top 5 results out of a combined
DataFrame of fifty Google and arXiv results. This will give us a good blend of 
diverse, specific, and relevant sources.
"""

df_google_twentyfive = web_search(WebSearchCorpus.GOOGLE, question, 25)[["title", "snippet"]]
df_arxiv_twentyfive = web_search(WebSearchCorpus.ARXIV, question, 25)[["title", "abstract"]]
df_arxiv_twentyfive = df_arxiv_twentyfive.rename(columns={"abstract": "snippet"})

df_twentyfive = pd.concat([df_google_twentyfive, df_arxiv_twentyfive])
"""
Use `sem_topk` to filter by relevance. We'll use the `quick` method, which allows us to quickly sort by relevance. 
We set `return_stats` to False because we do not require the specific statistics that were measured for scoring.
"""
df_twentyfive = df_twentyfive.sem_topk(
    "Which {snippet} is most relevant to the question and unique?", K=5, method="quick", return_stats=False
)

summarized_answer = df_twentyfive.sem_agg(
    "Summarize each {snippet} into a brief paragraph, answering the question: " + question
)._output[0]
print(f"Summarized results\n{summarized_answer}")

"""
A key advantage to using LOTUS, compared to LLM chats, is also that you possess all the data that you've used to answer the question.
We can continue using `df_twentyfive` to answer follow-up questionss, extracting different pieces of information or builiding on our
repository of context. Now, let's make use of the `parse_pdf` method to also pass in a textbook relevant to this question. 
"""


"""
Even with unusual formatting or interesting PDF layouts and over multiple pages, `parse_pdf`can accurately parse text rapdily. 
We can also simply pass in the URL of the PDF to parse it, so you do not even need to download the file.
"""
attention_is_all_you_need = parse_pdf("https://arxiv.org/pdf/1706.03762", per_page=False).rename(
    columns={"file_path": "title", "content": "snippet"}
)
the_transformer = parse_pdf("https://web.stanford.edu/~jurafsky/slp3/9.pdf", per_page=False).rename(
    columns={"file_path": "title", "content": "snippet"}
)

print(f"Parsed 'Attention is All You Need'\n{attention_is_all_you_need}")
print(f"Parsed Jurafsky's 'Speech and Language Processing'\n{the_transformer}")
df_combined = pd.concat([df_twentyfive, attention_is_all_you_need, the_transformer])

answer_with_textbooks = df_combined.sem_agg(
    "Summarize each {snippet} into a brief paragraph, answering the question: " + question
)._output[0]
print(f"Answer with textbooks\n{answer_with_textbooks}")

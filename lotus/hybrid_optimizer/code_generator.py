from typing import List

import re
import time
import ast
import litellm
import pandas as pd
import lotus.models
import logging

from .templates.prompt_templates import (
    KEYWORD_GENERATOR_SYSTEM_PROMPT,
    KEYWORD_GENERATOR_USER_PROMPT,
)
litellm.drop_params = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LIST_CONTENT_RE = re.compile(r'\[(.*?)\]', re.DOTALL)
TRAILING_COMMA_RE = re.compile(r',\s*\]')

class CodeGenerator:

    def generate_keywords(self, df: pd.DataFrame, instruction: str, example_data: str, model: lotus.models.LM, use_semantic_expansion: bool = False, num_keyword_calls: int = 1, **kwargs) -> List[str]:
        keywords_set = self._extract_keywords_from_llm(
            df=df, 
            instruction=instruction, 
            example_data=example_data, 
            model=model,
            num_calls=num_keyword_calls,
            **kwargs
        )
        return list(keywords_set)

    def _parse_llm_response_for_keywords(self, response_content: str) -> List[str]:
            list_match = LIST_CONTENT_RE.search(response_content)
            if not list_match:
                return []
            list_content_str = f"[{list_match.group(1)}]"


            list_content_str = list_content_str.replace("\\'", "'").replace('\\n', '') 
            list_content_str = TRAILING_COMMA_RE.sub(']', list_content_str).strip()

            if not list_content_str or list_content_str == '[]':
                return []

            keywords_parsed = ast.literal_eval(list_content_str)

            if keywords_parsed is None:
                return []

            if not isinstance(keywords_parsed, list):
                return []

            valid_keywords = []
            for k in keywords_parsed:
                if isinstance(k, (str, int, float)):
                    cleaned_k = str(k).lower().strip()
                    if cleaned_k:
                        valid_keywords.append(cleaned_k)
            
            return valid_keywords

    def _extract_keywords_from_llm(
        self, 
        df: pd.DataFrame, 
        instruction: str, 
        example_data: str, 
        model: lotus.models.LM, 
        temperature: float = 0.1, 
        num_calls: int = 3,
        max_retries_per_call: int = 2,
        temp_increment: float = 0.05
    ) -> set:
        
        enhanced_prompt = KEYWORD_GENERATOR_USER_PROMPT.format(
            instruction=instruction, example_data=example_data
        )
        
        model_name = model.model 

        litellm_kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": KEYWORD_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_prompt},
            ],
            "timeout": 120,
        }

        all_keywords = set() 

        for call_num in range(num_calls):
            current_temp = temperature + (call_num * temp_increment)
            call_kwargs = {**litellm_kwargs, "temperature": min(current_temp, 2.0)} 
            
            keywords_for_this_call = []
            for attempt in range(max_retries_per_call):
                try:
                    response = litellm.completion(**call_kwargs)
                    content = response.choices[0].message.content.strip()
                    
                    parsed_keywords = self._parse_llm_response_for_keywords(content)
                    
                    if parsed_keywords:
                        keywords_for_this_call = parsed_keywords
                        break

                except Exception as e:
                    if attempt == max_retries_per_call - 1:
                        raise e
                    else:
                        time.sleep(0.5)
            
            if keywords_for_this_call:
                all_keywords.update(keywords_for_this_call)

        return all_keywords

    # helper function to expand keywords if needed (not used in the current implementation)
    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        expanded_keywords = set(keywords)

        for keyword in keywords:
            expanded_keywords.add(keyword.lower())

            if " " in keyword:
                expanded_keywords.add(keyword.replace(" ", ""))
                expanded_keywords.add(keyword.replace(" ", "-"))
                expanded_keywords.add(keyword.replace(" ", "_"))

                words = keyword.split()
                if len(words) > 1:
                    acronym = "".join(word[0].lower() for word in words)
                    expanded_keywords.add(acronym)

                    if len(words) == 2:
                        compound = words[0][0].lower() + words[1].lower()
                        expanded_keywords.add(compound)

                    for i in range(len(words) - 1):
                        compound = words[i].lower() + words[i + 1].lower()
                        expanded_keywords.add(compound)

        return sorted(list(expanded_keywords))

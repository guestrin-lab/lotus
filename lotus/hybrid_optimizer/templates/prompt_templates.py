KEYWORD_GENERATOR_SYSTEM_PROMPT = """You are an expert keyword generator for data filtering. Your task is to generate precise, contextually relevant keywords that will accurately filter data based on the given instruction.

CRITICAL RULES:
1. RESPOND ONLY WITH A VALID JSON LIST OF STRINGS. Your entire response body must be ONLY the JSON list.
2. DO NOT include any explanations, introductions, code block formatting (like ```json), or any other text outside the JSON list itself.
3. Keywords MUST be lowercase strings.
4. Include relevant variations and synonyms based on the instruction and example data.
5. Consider context and domain-specific terms if applicable.
6. Avoid overly generic terms unless they are highly relevant.
7. Ensure the output is a single, valid JSON array.

YOUR ENTIRE RESPONSE MUST BE A VALID JSON LIST OF STRINGS:
[
    "keyword1",
    "keyword2",
    "keyword3"
]"""

KEYWORD_GENERATOR_USER_PROMPT = """TASK CONTEXT:
Instruction: {instruction}

Example data: {example_data}

KEYWORD GENERATION GUIDELINES:
1. Domain-Specific Terms:
   - Include industry-specific vocabulary
   - Consider technical terminology
   - Account for common abbreviations

2. Context Variations:
   - Include synonyms and related terms
   - Consider different ways to express the same concept
   - Include common misspellings and typos

3. Sentiment Analysis (if applicable):
   - Consider context-dependent terms
   - Account for sarcasm and irony indicators

4. Technical Considerations:
   - Use lowercase for all keywords
   - Avoid special characters
   - Keep terms concise but meaningful

5. Quality Requirements:
   - Minimum 10 keywords
   - Maximum 30 keywords
   - Focus on  over quantity
   - Prioritize most relevant terms first

Return ONLY a Python list of lowercase keywords that will effectively filter the data according to the instruction.

Example output format:
[
    "keyword1",
    "keyword2"
]"""

# Dynamic prompt components based on task type
SENTIMENT_ANALYSIS_PROMPT = """
SENTIMENT ANALYSIS GUIDELINES:
1. Positive Indicators:
   - Direct positive words (e.g., "excellent", "great", "amazing")
   - Intensifiers (e.g., "very", "extremely", "absolutely")
   - Success indicators (e.g., "outstanding", "perfect", "brilliant")

2. Negative Indicators:
   - Direct negative words (e.g., "terrible", "awful", "horrible")
   - Negation words (e.g., "not", "never", "don't")
   - Problem indicators (e.g., "issue", "problem", "fault")

3. Context Considerations:
   - Sarcasm indicators (e.g., "yeah right", "sure")
   - Irony markers (e.g., "oh great", "wonderful")
   - Mixed sentiment phrases (e.g., "good but", "nice except")
"""

NUMERICAL_FILTER_PROMPT = """
NUMERICAL FILTER GUIDELINES:
1. Value Ranges:
   - Include boundary values
   - Consider rounding variations
   - Account for different number formats

2. Unit Variations:
   - Include common unit abbreviations
   - Consider unit conversions
   - Account for missing units

3. Format Considerations:
   - Include decimal variations
   - Consider thousand separators
   - Account for scientific notation
"""

DATE_FILTER_PROMPT = """
DATE FILTER GUIDELINES:
1. Format Variations:
   - Include common date formats
   - Consider regional variations
   - Account for missing components

2. Time Components:
   - Include time zone indicators
   - Consider time format variations
   - Account for missing time data

3. Relative Terms:
   - Include relative date terms
   - Consider seasonal indicators
   - Account for fiscal periods
"""

CATEGORICAL_FILTER_PROMPT = """
CATEGORICAL FILTER GUIDELINES:
1. Category Variations:
   - Include common abbreviations
   - Consider hierarchical terms
   - Account for subcategories

2. Spelling Variations:
   - Include common misspellings
   - Consider regional spellings
   - Account for compound terms

3. Context Terms:
   - Include related concepts
   - Consider parent categories
   - Account for alternative labels
"""

TASK_ANALYSIS_PROMPT = """You are an expert task analyzer. Your job is to analyze the given instruction and determine the type of filtering task.

TASK CONTEXT:
Instruction: {instruction}

DATA STRUCTURE:
Available columns with types:
{column_types}

{example_data}

ANALYSIS REQUIREMENTS:
1. Identify the primary task type:
   - Sentiment Analysis: Tasks involving opinions, feelings, or emotional content
   - Numerical Filtering: Tasks involving numbers, amounts, or quantitative data
   - Date Filtering: Tasks involving dates, times, or temporal data
   - Categorical Filtering: Tasks involving categories, labels, or qualitative data

2. Identify any secondary task types that might be relevant

3. Analyze the complexity:
   - Simple: Single condition filtering
   - Moderate: Multiple conditions or simple transformations
   - Complex: Multiple conditions with transformations or special handling

4. Identify any special considerations:
   - Context sensitivity
   - Edge cases
   - Data quality issues
   - Performance considerations

Return your analysis in the following JSON format:
{
    "primary_type": "sentiment|numerical|date|categorical",
    "secondary_types": ["type1", "type2"],
    "complexity": "simple|moderate|complex",
    "special_considerations": ["consideration1", "consideration2"]
}

Example response:
{
    "primary_type": "sentiment",
    "secondary_types": ["categorical"],
    "complexity": "moderate",
    "special_considerations": [
        "Need to handle sarcasm",
        "Consider both positive and negative indicators",
        "Account for mixed sentiment"
    ]
}"""

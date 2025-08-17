KEYWORD_FILTER_TEMPLATE = """def hybrid_filter(df):
    keywords = {keywords}
    
    import numpy as np
    from numpy.char import lower as np_lower
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    def process_chunk(chunk_data, keywords_arr):
        chunk = np_lower(chunk_data.astype(str))
        chunk_reshaped = chunk.reshape(-1, 1)
        keywords_reshaped = keywords_arr.reshape(1, -1)
        return np.any(np.char.find(chunk_reshaped, keywords_reshaped) >= 0, axis=1)
    
    mask = np.zeros(len(df), dtype=bool)
    
    min_length = 3
    keywords_arr = np.array([
        k for k in keywords 
        if len(k) >= min_length and not k.isspace()
    ], dtype=str)
    
    chunk_size = min(10000, max(1000, len(df) // (4 * len(keywords_arr))))
    
    for col in {relevant_cols}:
        if df[col].dtype == 'object':
            col_data = df[col].fillna('')
            n_rows = len(col_data)
            
            col_array = col_data.to_numpy()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunks = [
                    col_array[i:min(i + chunk_size, n_rows)]
                    for i in range(0, n_rows, chunk_size)
                ]
                process_func = partial(process_chunk, keywords_arr=keywords_arr)
                results = executor.map(process_func, chunks)
                
                start_idx = 0
                for chunk_result in results:
                    end_idx = min(start_idx + chunk_size, n_rows)
                    mask[start_idx:end_idx] |= chunk_result
                    start_idx = end_idx
    
    return df[mask]"""

KEYWORD_FILTER_TEMPLATE_LAMBDA = """(lambda: (
    lambda df, kw={keywords}, cols={relevant_cols}, ml=3: (
        lambda np, pd, re, ThreadPoolExecutor: (
            df.copy() if not kw or df.empty else (
                df.copy() if not (cols := cols or df.select_dtypes(include=['object']).columns.tolist()) else (
                    df.copy() if not (valid_kw := [k.lower() for k in kw if isinstance(k, str) and len(k) >= ml and not k.isspace()]) else (
                        df[df.apply(lambda row: any(k in str(v).lower() for k in valid_kw for v in (row[cols] if cols else row)), axis=1)]
                    )
                )
            )
        )
    )(
        __import__('numpy'),
        __import__('pandas'),
        __import__('re'),
        __import__('concurrent.futures').ThreadPoolExecutor
    )
))()"""

KEYWORD_FILTER_TEMPLATE_OPTIMIZED = """def hybrid_filter(df, kw={keywords}, cols={relevant_cols}, ml=3):
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import re
    
    if not kw or df.empty: 
        return df.copy()
    
    if not cols: 
        cols = df.select_dtypes(include=['object']).columns.tolist()
    if not cols:
        return df.copy()
    
    valid_kw = [k.lower() for k in kw if isinstance(k, str) and len(k) >= ml and not k.isspace()]
    if not valid_kw:
        return df.copy()
    
    pattern = re.compile('|'.join(map(re.escape, valid_kw)))
    
    def search_col_optimized(col_series):
        col_data = col_series.fillna('').astype(str)
        col_data = col_data.str.lower()
        return col_data.str.contains(pattern, regex=True)
    
    def process_chunk(chunk_df, chunk_cols):
        chunk_mask = np.zeros(len(chunk_df), dtype=bool)
        for c in chunk_cols:
            matches = search_col_optimized(chunk_df[c])
            chunk_mask |= matches
            if chunk_mask.all():
                break
        return chunk_mask
    
    df_size = len(df)
    use_chunking = df_size > 100_000
    
    if not use_chunking:
        mask = np.zeros(df_size, dtype=bool)
        for col in cols:
            if mask.all():
                break
            matches = search_col_optimized(df[col])
            mask |= matches
        return df[mask]
    else:
        chunk_size = min(10_000, max(1_000, df_size // 10))
        num_chunks = (df_size + chunk_size - 1) // chunk_size
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
        
        with ThreadPoolExecutor(max_workers=min(8, num_chunks)) as executor:
            results = list(executor.map(
                lambda chunk: process_chunk(chunk, cols),
                chunks
            ))
        
        final_mask = np.concatenate(results)
        return df[final_mask]"""

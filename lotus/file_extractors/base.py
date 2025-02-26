import tempfile
from pathlib import Path

import pandas as pd

import lotus


def get_custom_readers(custom_reader_configs: dict[str, dict] | None = None):
    if custom_reader_configs is None:
        custom_reader_configs = {}

    if not isinstance(custom_reader_configs, dict):
        raise ValueError("custom_reader_configs must be a dictionary")

    from .pptx import PptxReader

    pptx_custom_reader = PptxReader(**custom_reader_configs.get("pptx", {}))
    return {
        ".pptx": pptx_custom_reader,
        ".ppt": pptx_custom_reader,
        ".pptm": pptx_custom_reader,
    }


def load_files(
    file_paths: list[str] | str | Path | list[Path],
    recursive: bool = False,
    per_page: bool = True,
    page_separator: str = "\n",
    custom_reader_configs: dict[str, dict] | None = None,
    request_timeout: int | None = None,
    show_progress: bool = False,
    num_workers: int | None = None,
) -> pd.DataFrame:
    """
    Load files and return the content in a DataFrame.

    Args:
        file_paths (list[str] | str | Path | list[Path]): A list of file paths/urls or a single file path/url. File path can be directory path as well in which case all files in the directory will be loaded.
        recursive (bool): If True, load files from subdirectories as well for directories in file_paths. Else, only load files from the specified directories. Default is False.
        per_page (bool): If True, return the content of each page as a separate row if the document has multiple pages. Default is True.
        page_separator (str): The separator to use when joining the content of each page in case per_page is False. Default is "\n".
        custom_reader_configs (dict): A dictionary containing configurations for custom readers. The key should be the file extension and the value should be a dictionary containing the configurations for the custom reader.
        request_timeout (int): The number of seconds to wait if fetching a file from a URL. Default is None.
        num_workers (int): The number of workers to use for loading files. Default is None.
        show_progress (bool): If True, show a progress bar while loading files. Default is False.
    """
    try:
        from .directory_reader import DirectoryReader
    except ImportError:
        raise ImportError(
            "The 'llama-index' and 'python-magic' library is required for parsing documents. "
            "You can install it with the following command:\n\n"
            "    pip install llama-index python-magic"
        )
    if isinstance(file_paths, str) or isinstance(file_paths, Path):
        file_paths = [file_paths]  # type: ignore

    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list of file paths/urls.")

    with tempfile.TemporaryDirectory() as temp_dir:
        directory_reader = DirectoryReader(
            recursive=recursive, file_extractor=get_custom_readers(custom_reader_configs)
        )
        for file_path in file_paths:
            try:
                directory_reader.add(file_path, temp_dir, request_timeout)
            except Exception as e:
                lotus.logger.error(f"Failed to load {file_path}. Error: {e}. Skipping...")

        llamaindex_documents = directory_reader.load_data(
            per_page=per_page, show_progress=show_progress, page_separator=page_separator, num_workers=num_workers
        )
        all_data = [{"content": doc.text, **doc.metadata} for doc in llamaindex_documents]
        df = pd.DataFrame(all_data)

    return df

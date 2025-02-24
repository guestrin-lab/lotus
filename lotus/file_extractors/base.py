import mimetypes
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests  # type: ignore

import lotus


def get_custom_readers(custom_reader_configs: dict[str, dict] = {}):
    from .pptx import PptxReader

    pptx_custom_reader = PptxReader(custom_reader_configs.get("pptx", {}))
    return {
        ".pptx": pptx_custom_reader,
        ".ppt": pptx_custom_reader,
        ".pptm": pptx_custom_reader,
    }


def is_url(path: str | Path) -> bool:
    try:
        result = urlparse(str(path))
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_extension(content):
    import magic

    mime = magic.Magic(mime=True).from_buffer(content) or "application/octet-stream"
    return mimetypes.guess_extension(mime) or ".bin"


def load_files(
    file_paths: list[str] | str | Path | list[Path],
    recursive: bool = False,
    per_page: bool = True,
    page_separator: str = "\n",
    custom_reader_configs: dict[str, dict] = {},
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Load files and return the content in a DataFrame.

    Args:
        file_paths (list[str] | str | Path | list[Path]): A list of file paths/urls or a single file path/url. File path can be directory path as well in which case all files in the directory will be loaded.
        recursive (bool): If True, load files from subdirectories as well for directories in file_paths. Else, only load files from the specified directories. Default is False.
        per_page (bool): If True, return the content of each page as a separate row if the document has multiple pages. Default is True.
        page_separator (str): The separator to use when joining the content of each page in case per_page is False. Default is "\n".
        custom_reader_configs (dict): A dictionary containing configurations for custom readers. The key should be the file extension and the value should be a dictionary containing the configurations for the custom reader.
        show_progress (bool): If True, show a progress bar while loading files. Default is False.
    """
    try:
        from .directory_reader import DirectoryReader
    except ImportError:
        raise ImportError(
            "The 'llama-index' library is required for parsing documents. "
            "You can install it with the following command:\n\n"
            "    pip install 'llama-index'"
        )
    if isinstance(file_paths, str) or isinstance(file_paths, Path):
        file_paths = [file_paths]  # type: ignore
    assert isinstance(file_paths, list), "file_paths must be a list of file paths/urls."

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path_mappings = {}
        directory_reader = DirectoryReader(
            recursive=recursive, filename_as_id=True, file_extractor=get_custom_readers(custom_reader_configs)
        )
        for file_path in file_paths:
            if is_url(file_path):
                response = requests.get(file_path)
                if response.status_code != 200:
                    lotus.logger.error(
                        f"Failed to download file from {file_path}. Status code: {response.status_code}. Error Message: {response.text}. Skipping..."
                    )
                    continue
                _file_path = tempfile.NamedTemporaryFile(
                    delete=False, dir=temp_dir, suffix=get_extension(response.content)
                ).name
                file_path_mappings[_file_path] = file_path
                with open(_file_path, "wb") as f:
                    f.write(response.content)
                directory_reader.add_file(_file_path)
            elif Path(file_path).is_file():
                directory_reader.add_file(file_path)
            elif Path(file_path).is_dir():
                directory_reader.add_dir(file_path)
            else:
                lotus.logger.error(f"{file_path} is not a valid file or directory. Skipping...")

        all_data = []
        for docs in directory_reader.iter_data(show_progress):
            if len(docs) > 1 and not per_page:
                metadata = docs[0].metadata
                metadata.pop("page_label", None)
                metadata["file_path"] = file_path_mappings.get(metadata.get("file_path"), metadata.get("file_path"))
                all_data.append({"content": page_separator.join([doc.text for doc in docs]), **metadata})
            else:
                for doc in docs:
                    doc.metadata["file_path"] = file_path_mappings.get(
                        doc.metadata.get("file_path"), doc.metadata.get("file_path")
                    )
                    doc.metadata["page_label"] = int(doc.metadata.get("page_label", 1))
                all_data.extend([{"content": doc.text, **doc.metadata} for doc in docs])
        df = pd.DataFrame(all_data)

    return df

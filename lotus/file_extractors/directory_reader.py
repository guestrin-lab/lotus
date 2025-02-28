import mimetypes
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Generator
from urllib.parse import urlparse

import fsspec
import magic
import pandas as pd
import requests  # type: ignore
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document, SimpleDirectoryReader


def get_path_class(fs: fsspec.AbstractFileSystem, file_path: str | Path | PurePosixPath) -> Path | PurePosixPath:
    """Check if filesystem is the default local filesystem."""
    is_default_fs = isinstance(fs, LocalFileSystem) and not fs.auto_mkdir
    return Path(file_path) if is_default_fs else PurePosixPath(file_path)


def get_extension(content: bytes) -> str:
    """Determine file extension from content using magic library."""
    try:
        mime = magic.Magic(mime=True).from_buffer(content) or "application/octet-stream"
        return mimetypes.guess_extension(mime) or ".bin"
    except Exception:
        return ".bin"  # Fallback extension if detection fails


def is_url(path: str | Path) -> bool:
    """Check if a given path is a valid URL."""
    try:
        result = urlparse(str(path))
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


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


class DirectoryReader:
    """
    Enhanced wrapper on SimpleDirectoryReader allowing incremental addition of files.

    This class extends SimpleDirectoryReader functionality by supporting:
    - Incremental file/directory addition
    - URL downloads with automatic file type detection
    - Cleanup of temporary files
    - Convenient unified interface for adding content

    Args:
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        custom_reader_configs (dict): A dictionary containing configurations for custom readers. The key should be the file extension and the value should be a dictionary containing the configurations for the custom reader.

    FROM SimpleDirectoryReader:
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        exclude_empty (bool): Whether to exclude empty files (Optional).
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
            Default is None.
        raise_on_error (bool): Whether to raise an error if a file cannot be read.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
        to using the local file system. Can be changed to use any remote file system
        exposed via the fsspec interface.
    """

    def __init__(self, recursive: bool = False, custom_reader_configs: dict[str, dict] | None = None, **kwargs):
        self.reader = None
        self.temp_file_to_url_map: dict[str, str] = {}
        kwargs["filename_as_id"] = True  # need to set this to True for proper metadata handling
        self.reader_kwargs = {
            **kwargs,
            "recursive": recursive,
            "file_extractor": get_custom_readers(custom_reader_configs),
        }

    def add_file(self, file_path: str | Path) -> "DirectoryReader":
        """
        Add a single file to the reader.

        Args:
            file_path: Path to the file

        Returns:
            DirectoryReaderobject: To allow chaining of methods

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if self.reader is None:
            self.reader = SimpleDirectoryReader(input_files=[file_path], **self.reader_kwargs)
            return self

        # Verify file exists
        if not self.reader.fs.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.reader.input_files.append(get_path_class(self.reader.fs, file_path))

    def add_dir(self, input_dir: str | Path) -> "DirectoryReader":
        """
        Add a directory to the reader.

        Args:
            input_dir: Path to the directory

        Returns:
            DirectoryReaderobject: To allow chaining of methods

        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        if self.reader is None:
            self.reader = SimpleDirectoryReader(input_dir=input_dir, **self.reader_kwargs)
            return self

        # Verify directory exists
        if not self.reader.fs.isdir(input_dir):
            raise FileNotFoundError(f"Directory {input_dir} does not exist.")
        self.reader.input_files.extend(self.reader._add_files(get_path_class(self.reader.fs, input_dir)))

        return self

    def add_url(self, url: str | Path, temp_dir: str | None = None, timeout: int | None = None) -> "DirectoryReader":
        """
        Download and add a file from a URL.

        Args:
            url: URL to the file
            temp_dir: Optional temporary directory to store downloaded files
            timeout: Optional timeout for the HTTP request in seconds

        Returns:
            DirectoryReaderobject: To allow chaining of methods

        Raises:
            ValueError: If download or processing fails
        """
        _file_path = None
        try:
            # Using stream mode to allow large files
            with requests.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()

                # Download initial chunk to determine file type
                content = b""
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > 2048:
                        break

                # Create temporary file
                _file_path = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=get_extension(content)).name

                # Write content to file
                with open(_file_path, "wb") as f:
                    # Write what we've already downloaded
                    f.write(content)
                    # Continue downloading if there's more content
                    if len(response.content) > len(content):
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:  # filter out keep-alive chunks
                                continue
                            f.write(chunk)

            self.add_file(_file_path)
            self.temp_file_to_url_map[_file_path] = str(url)

        except Exception as e:
            if _file_path and Path(_file_path).exists():
                Path(_file_path).unlink()
            raise ValueError(f"Failed to process file from URL {url}. Error: {e}")

        return self

    def add(self, path: str | Path, temp_dir: str | None = None, timeout: int | None = None) -> "DirectoryReader":
        """
        Universal method to add a file, directory, or URL to the reader.

        Args:
            path: URL or Path to file/directory
            temp_dir: Optional temporary directory for URL downloads
            timeout: Optional timeout for URL requests in seconds

        Returns:
            DirectoryReaderobject: To allow chaining of methods

        Raises:
            ValueError: If path is invalid or processing fails
        """
        if is_url(path):
            self.add_url(path, temp_dir, timeout)
        elif Path(path).is_file():
            self.add_file(path)
        elif Path(path).is_dir():
            self.add_dir(path)
        else:
            raise ValueError(f"{path} is not a valid file, directory, or URL.")

        return self

    def add_multiple(
        self, paths: list[str | Path], temp_dir: str | None = None, timeout: int | None = None
    ) -> "DirectoryReader":
        """
        Add multiple files, directories, or URLs to the reader.

        Args:
            paths: List of URLs or Paths to files/directories
            temp_dir: Optional temporary directory for URL downloads
            timeout: Optional timeout for URL requests in seconds

        Returns:
            DirectoryReaderobject: To allow chaining of methods

        Raises:
            ValueError: If any path is invalid or processing fails
        """
        for path in paths:
            self.add(path, temp_dir, timeout)

        return self

    def _process_metadata(self, docs: list[Document], add_page_label: bool) -> Document:
        for doc in docs:
            if doc.metadata.get("file_path") in self.temp_file_to_url_map:
                doc.metadata["file_path"] = self.temp_file_to_url_map[doc.metadata["file_path"]]
            if add_page_label:
                doc.metadata["page_label"] = int(doc.metadata.get("page_label", 1))
            else:
                doc.metadata.pop("page_label", None)
        return docs

    def iter_data(
        self, per_page: bool = True, page_separator: str = "\n", show_progress: bool = False
    ) -> Generator[list[Document], Any, Any]:
        """
        Iterate over the loaded documents.

        Args:
            per_page: Whether to return each page as a separate document
            show_progress: Whether to show a progress bar

        Yields:
            Lists of Document objects
        """
        if self.reader is None:
            raise ValueError("No files, directories, or URLs have been added.")

        for data in self.reader.iter_data(show_progress=show_progress):
            self._process_metadata(data, per_page)
            if not per_page:
                yield [Document(text=page_separator.join([doc.text for doc in data]), metadata=data[0].metadata)]
            yield data

    def load_data(
        self,
        per_page: bool = True,
        page_separator: str = "\n",
        show_progress: bool = False,
        num_workers: int | None = None,
    ) -> list[Document]:
        """
        Load all documents at once.

        Args:
            per_page: Whether to return each page as a separate document
            show_progress: Whether to show a progress bar
            num_workers: Number of workers to use for parallel processing

        Returns:
            List of all Document objects
        """
        if self.reader is None:
            raise ValueError("No files, directories, or URLs have been added.")

        docs = self.reader.load_data(show_progress=show_progress, num_workers=num_workers)
        self._process_metadata(docs, per_page)
        if not per_page:
            grouped_docs: defaultdict[str, list[Document]] = defaultdict(list)
            for doc in docs:
                grouped_docs[doc.metadata.get("file_name")].append(doc)
            merged_docs = [
                Document(text=page_separator.join([doc.text for doc in group]), metadata=group[0].metadata)
                for group in grouped_docs.values()
            ]
            return merged_docs
        return docs

    def to_df(
        self,
        per_page: bool = True,
        page_separator: str = "\n",
        show_progress: bool = False,
        num_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        Load files and return the content in a DataFrame.

        Args:
            per_page (bool): If True, return the content of each page as a separate row if the document has multiple pages. Default is True.
            page_separator (str): The separator to use when joining the content of each page in case per_page is False. Default is "\n".
            num_workers (int): The number of workers to use for loading files. Default is None.
            show_progress (bool): If True, show a progress bar while loading files. Default is False.
        """
        llamaindex_documents = self.load_data(
            per_page=per_page, show_progress=show_progress, page_separator=page_separator, num_workers=num_workers
        )
        all_data = [{"content": doc.text, **doc.metadata} for doc in llamaindex_documents]
        return pd.DataFrame(all_data)

    def __del__(self) -> None:
        """Automatically clean up temporary files when the reader is garbage collected."""
        for temp_file in list(self.temp_file_to_url_map.keys()):
            if Path(temp_file).exists():
                try:
                    print(f"Deleting temporary file: {temp_file}")
                    Path(temp_file).unlink()
                    del self.temp_file_to_url_map[temp_file]
                except Exception:
                    pass  # Silently continue if deletion fails

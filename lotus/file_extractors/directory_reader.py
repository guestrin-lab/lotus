from pathlib import Path, PurePosixPath

import fsspec
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import SimpleDirectoryReader


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


class DirectoryReader:
    """
    Creating a wrapper on SimpleDirectoryReader to allow incremental addition of files.

    Args:
        input_dir (Union[Path, str]): Path to the directory.
        input_files (List): List of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        exclude_empty (bool): Whether to exclude empty files (Optional).
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        filename_as_id (bool): Whether to use the filename as the document id.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
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

    def __init__(self, **kwargs):
        self.reader = None
        self.reader_kwargs = kwargs

    def add_file(self, file_path: str | Path):
        """
        Add a file to the reader.

        Args:
            file_path (str): Path to the file.
        """
        if self.reader is None:
            self.reader = SimpleDirectoryReader(input_files=[file_path], **self.reader_kwargs)
        else:
            if not self.reader.fs.isfile(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist.")
            _Path = Path if is_default_fs(self.fs) else PurePosixPath
            self.reader.input_files.append(_Path(file_path))

    def add_dir(self, input_dir: str | Path):
        """
        Add a directory to the reader.

        Args:
            input_dir (str): Path to the directory.
        """
        if self.reader is None:
            self.reader = SimpleDirectoryReader(input_dir=input_dir, **self.reader_kwargs)
        else:
            if not self.reader.fs.isdir(input_dir):
                raise FileNotFoundError(f"Directory {input_dir} does not exist.")
            _Path = Path if is_default_fs(self.fs) else PurePosixPath
            self.reader.input_files.extend(self.reader._add_files(_Path(input_dir)))

    def iter_data(self, show_progress: bool = False):
        if self.reader is None:
            raise ValueError("No files or directories have been added.")
        return self.reader.iter_data(show_progress)

    def __getattribute__(self, name):
        if name not in ["add_file", "add_dir", "reader", "reader_kwargs", "iter_data"]:
            if self.reader:
                return getattr(self.reader, name)
            else:
                raise AttributeError(f"'DirectoryReader' object has no attribute '{name}'")
        else:
            return super().__getattribute__(name)

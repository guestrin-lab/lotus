import pandas as pd

import lotus
from lotus.file_extractors import DirectoryReader

################################################################################
# Setup
################################################################################
# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")


def test_parse_pdf():
    pdf_urls = ["https://arxiv.org/pdf/1706.03762", "https://arxiv.org/pdf/2407.11418"]

    df = DirectoryReader().add_multiple(pdf_urls).to_df(per_page=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["file_path"].tolist() == pdf_urls


def test_parse_pdf_per_page():
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    df = DirectoryReader().add(pdf_url).to_df(per_page=True)

    assert isinstance(df, pd.DataFrame)

    # Check if the content is split into pages and the page numbers are correct
    assert len(df) == 15
    assert sorted(df["page_label"].unique()) == list(range(1, 16))

    # Check if all rows have the filepath set to the URL
    assert all(df["file_path"] == pdf_url)

Installation
============

LOTUS can be installed as a Python library through pip.

Requirements
------------

* OS: MacOS, Linux
* Python: 3.10

Install with pip
----------------

You can install LOTUS using pip:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate lotus
    $ pip install lotus-ai

If you are running on mac, please install Faiss via conda:

.. code-block:: console

    # CPU-only version
    $ conda install -c pytorch faiss-cpu=1.8.0

    # GPU(+CPU) version
    $ conda install -c pytorch -c nvidia faiss-gpu=1.8.0

For more details, see `Installing FAISS via Conda <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#installing-faiss-via-conda>`_.

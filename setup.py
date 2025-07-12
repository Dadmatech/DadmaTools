import setuptools
import os
import sys


if "google.colab" not in sys.modules:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "certifi>=2025.4.26,<2026.0.0",
    "charset-normalizer>=3.4.2,<4.0.0",
    "idna>=3.10,<4.0",
    "urllib3>=2.4.0,<3.0",
    "requests>=2.32.3,<3.0",
    "emoji==2.14.1",
    "click>=8.2.1,<9.0",
    "colorama>=0.4.6,<1.0",
    "tqdm>=4.67.1,<5.0",
    "python-dateutil==2.9.0.post0",
    "pytz>=2025.2,<2026.0",
    "tzdata>=2025.2,<2026.0",
    "PyYAML>=6.0.2,<7.0",
    "fsspec>=2025.3.2,<2025.6.0",
    "filelock>=3.18.0,<4.0",
    "numpy>=1.24.0,<2.7.0",
    "pandas>=2.2.2",
    "scipy>=1.11.0,<2.0",
    "joblib>=1.5.0,<2.0",
    "networkx>=3.4.2,<4.0",
    "regex>=2024.11.6,<2025.0",
    "gensim>=4.3.2,<5.0",
    "conllu>=4.7.3,<7.0",
    "langid>=1.1.6,<2.0",
    "sacremoses>=0.1.1,<1.0",
]

core_extra = [
    "protobuf>=3.20.3,<6.0.0",
    "sentencepiece>=0.1.99,<0.2.0",
    "transformers>=4.30.0,<5.0",
    "tokenizers==0.21.1",
    "sentence-transformers>=2.2.2,<3.0",
    "torch>=2.2.0,<2.7.0",
    "torchtext>=0.17.2,<0.19.0",
    "safetensors>=0.5.3,<1.0",
]

full_extra = [
    "brotli>=1.1.0,<2.0",
    "inflate64>=1.0.1,<2.0",
    "pyzstd>=0.17.0,<1.0",
    "threadpoolctl>=3.6.0,<4.0",
    "huggingface-hub>=0.31.4,<1.0",
    "scikit-learn>=1.6.1,<2.0",
    "psutil>=5.9.0,<8.0",
    "sympy>=1.13.1,<2.0",
    "mpmath>=1.3.0,<2.0",
    "html2text>=2025.4.15,<2026.0",
    "gdown>=5.2.0,<6.0",
    "py7zr>=0.22.0,<0.23.0",
    "pybcj>=1.0.6,<2.0",
    "pycryptodomex>=3.23.0,<4.0",
    "pyppmd>=1.1.1,<2.0",
    "texttable>=1.7.0,<2.0",
    "multivolumefile>=0.2.3,<0.3.0",
    "tsfresh>=0.20.0,<1.0",
]

colab_extra = [
    "transformers>=4.30.0,<5.0",
    "sentencepiece>=0.1.99,<0.2.0",
    "torchtext>=0.17.2,<0.19.0",
]

setuptools.setup(
    name="dadmatools",
    version="2.3.4",
    author="Dadmatech AI Company",
    author_email="info@dadmatech.ir",
    description="DadmaTools is a Persian NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dadmatech/DadmaTools",
    packages=setuptools.find_packages(),
    python_requires=">=3.11,<3.13",
    install_requires=install_requires,
    extras_require={
        "full": core_extra + full_extra,
        "colab": colab_extra,
    },
    dependency_links=[
        "git+https://github.com/kpu/kenlm@master#egg=kenlm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

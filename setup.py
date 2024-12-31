import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Heavy dependencies
heavy_dependencies = [
    "torch>=2.3.1",
    "transformers>=4.9.1",
    "pytorch-transformers>=1.1.0",
    "tf-estimator-nightly==2.8.0.dev2021122109",
    "supar==1.1.2",
]

# Base dependencies for lightweight installation
base_dependencies = [
    "bpemb>=0.3.3",
    "nltk",
    "folium>=0.2.1",
    "h5py>=3.3.0",
    "Deprecated==1.2.6",
    "hyperopt>=0.2.5",
    "pyconll>=3.1.0",
    "segtok>=1.5.7",
    "tabulate>=0.8.6",
    "gensim>=3.6.0",
    "conllu",
    "gdown>=4.3.1",
    "py7zr>=0.17.2",
    "html2text",
    "scikit-learn>=0.24.2",
    "numpy",
    "protobuf",
    "requests",
    "tqdm>=4.27",
    "langid==1.1.6",
    "filelock",
    "tokenizers>=0.7.0",
    "regex != 2019.12.17",
    "packaging",
    "sentencepiece",
    "sacremoses",
    "fasttext-wheel",
    "kenlm",
    "emoji==2.10.0",
    "pandas",
]

# Combine base and heavy dependencies for the default installation
default_dependencies = base_dependencies + heavy_dependencies

setuptools.setup(
    name="dadmatools",
    version="2.1.4",
    author="Dadmatech AI Company",
    author_email="info@dadmatech.ir",
    description="DadmaTools is a Persian NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dadmatech/DadmaTools",
    packages=setuptools.find_packages(),
    install_requires=default_dependencies,  # Default to all dependencies
    extras_require={
        "light": base_dependencies,  # Lightweight installation option
    },
    dependency_links=[
        "git+https://github.com/kpu/kenlm@master#egg=kenlm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License ",
        "Operating System :: OS Independent",
    ],
)

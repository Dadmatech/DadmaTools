import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dadmatools",
    version="2.0.2",
    author="Dadmatech AI Company",
    author_email="info@dadmatech.ir",
    description="DadmaTools is a Persian NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dadmatech/DadmaTools",
    packages=setuptools.find_packages(),
    install_requires=[
        "bpemb>=0.3.3",
        "nltk",
        "folium>=0.2.1",
        "spacy>=3.0.0",
        "torch>=1.7.1",
        "transformers>=4.9.1",
        "h5py>=3.3.0",
        "Deprecated==1.2.6",
        "hyperopt>=0.2.5",
        "pyconll>=3.1.0",
        "pytorch-transformers>=1.1.0",
        "segtok>=1.5.7",
        "tabulate>=0.8.6",
        "supar==1.1.2",
        "gensim>=3.6.0",
        "conllu",
        "gdown>=4.3.1",
        "py7zr>=0.17.2",
        "html2text",
        "tf-estimator-nightly==2.8.0.dev2021122109",
        "scikit-learn>=0.24.2",
        'numpy',
        'protobuf',
        'requests',
        'tqdm>=4.27',
        'langid==1.1.6',
        'filelock',
        'tokenizers>=0.7.0',
        'regex != 2019.12.17',
        'packaging',
        'sentencepiece',
        'sacremoses',
        'fasttext',
        'kenlm',
    ],

    dependency_links = [
        'git+https://github.com/kpu/kenlm@master#egg=kenlm',
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License ",
        "Operating System :: OS Independent",
    ],
)

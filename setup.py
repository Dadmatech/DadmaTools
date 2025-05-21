import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Heavy dependencies
heavy_dependencies = [
    "torch==2.1.0",
    "transformers==4.47.1",
    "pytorch-transformers==1.2.0",
    "tensorflow-estimator==2.8.0",
    "supar==1.1.2",
]

# Base dependencies for lightweight installation
base_dependencies = [
    "bpemb==0.3.6",
    "nltk==3.9.1",
    "folium==0.19.3",
    "h5py==3.12.1",
    "Deprecated==1.2.6",
    "hyperopt==0.2.7",
    "pyconll==3.2.0",
    "segtok>=1.5.7",
    "tabulate==0.9.0",
    "gensim==4.3.3",
    "conllu==6.0.0",
    "gdown==5.2.0",
    "py7zr==0.22.0",
    "html2text==2024.2.26",
    "scikit-learn==1.6.0",
    "numpy==1.26.4",
    "protobuf==5.29.2",
    "requests==2.32.3",
    "tqdm==4.67.1",
    "langid==1.1.6",
    "filelock==3.16.1",
    "tokenizers==0.21.0",
    "regex==2024.11.6",
    "packaging==24.2",
    "sentencepiece==0.2.0",
    "sacremoses==0.1.1",
    "emoji==2.10.0",
    "pandas==2.2.3",
    "py3langid==0.3.0"
]

# Combine base and heavy dependencies for the default installation
default_dependencies = base_dependencies + heavy_dependencies

setuptools.setup(
    name="dadmatools",
    version="2.1.8",
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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)

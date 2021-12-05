import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dadmatools",
    version="1.2.3",
    author="Dadmatech AI Company",
    author_email="info@dadmatech.ir",
    description="DadmaTools is a Persian NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dadmatech/DadmaTools",
    packages=setuptools.find_packages(),
    install_requires=[
	"bpemb==0.3.3",
	"spacy==3.0.0",
    	"sklearn==0.0",
	"torch==1.7.1",
	"transformers==4.9.1",
	"ipython==7.12.0",
	"h5py==3.3.0",
	"Deprecated==1.2.6",
	"hyperopt==0.2.5",
	"langdetect==1.0.9",
	"matplotlib-inline==0.1.2",
	"mpld3==0.3",
	"pyconll==3.1.0",
	"pyhocon==0.3.56",
	"pytorch-transformers==1.1.0",
	"segtok==1.5.7",
	"sqlitedict==1.7.0",
	"tabulate==0.8.6",
	"typing-utils==0.1.0",
	"supar==1.1.2"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License ",
        "Operating System :: OS Independent",
    ],
)

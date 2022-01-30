import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dadmatools",
    version="1.3.8",
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
	"folium==0.2.1",
	"spacy==3.0.0",
    	"sklearn>=0.0",
	"torch>=1.7.1",
	"transformers==4.9.1",
	"h5py>=3.3.0",
	"Deprecated==1.2.6",
	"hyperopt>=0.2.5",
	"pyconll>=3.1.0",
	"pytorch-transformers>=1.1.0",
	"segtok==1.5.7",
	"tabulate>=0.8.6",
	"supar>=1.1.2",
	"html2text",
	"gensim>=3.6.0",
	"fasttext==0.9.2",
	"wiki_dump_reader",
	"conllu",
	"gdown",
	"NERDA",
	"py7zr==0.17.2"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License ",
        "Operating System :: OS Independent",
    ],
)

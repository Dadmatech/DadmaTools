Dependency Parsing & Noun Phrase Chunking
=========================================

## Dependency Parsing

Dependency parsing is the task of extracting a dependency parse of a sentence. 
It is typically represented by a directed graph that depicts the grammatical structure of the sentence; where nodes are words and edges define syntactic relations between those words. 
A dependency relation is a triplet consisting of: a head (word), a dependent (another word) and a dependency label (describing the type of the relation).


## Noun Phrase Chunking

Chunking is the task of grouping words of a sentence into syntactic phrases (e.g. noun-phrase, verb phrase). 
Here, we focus on the prediction of noun-phrases (NP). Noun phrases can be pronouns (`PRON`), proper nouns (`PROPN`) or nouns (`NOUN`)  -- potentially bound with other tokens that act as modifiers, e.g., adjectives (`ADJ`) or other nouns. 

## How to use

Dependency parsing and chunking can be used seperately as preprocessing steps for other NLP tasks. If you want to use dependacy parsing only just add `dep` in the pipeline and if you want to use only chunking just add `chunk` in the pipeline. If you wish to use both add `dep,chunk` to pipeline.




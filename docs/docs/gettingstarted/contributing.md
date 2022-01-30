Contributing
============

If you want to contribute to the DadmaTools project, your help is very welcome. You can contribute to the project in many ways:

- Help us write good tutorials on Persian NLP use-cases.

- Contribute with your own pretrained NLP models, embeddings, or datasets in Persian.
You can add also your own pipeline using `add_pipe` when you have created `dadmatools.pipeline.language.pipeline`.

```python
import dadmatools.pipeline.language as language
pips = '<choose whatever you want for pipelion>' 
nlp = language.Pipeline(pips)
nlp.add_pipe('<your own OR spaCy default pipeline>')
```

- Create GitHub issues with questions and bug reports.

- Notify us of other Persian NLP resources or tell us about any good ideas that you have for improving the project through the [Discussions](https://github.com/Dadmatech/DadmaTools/discussions) section of the GitHub repository.
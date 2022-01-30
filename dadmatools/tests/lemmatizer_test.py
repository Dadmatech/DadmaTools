import dadmatools.models.lemmatizer as lem


if __name__ == '__main__':
    tokens = [['از', 'قصهٔ', 'کودکی', 'شان' ,'که', 'می‌گفت', '،', 'گاهی', 'حرص', 'می‌خورد']]

    model, args = lem.load_model()
    results = lem.lemma(model, args, tokens)
    print(results)
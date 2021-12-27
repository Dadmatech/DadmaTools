import dadmatools.models.tokenizer as tok

if __name__ == '__main__':
    text = 'از قصه کودکیشان که می‌گفت، گاهی حرص می‌خورد'

    model, args = tok.load_model()
    results = tok.tokenizer(model, args, text)
    print(results)
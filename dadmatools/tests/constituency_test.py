import dadmatools.models.constituency_parser as cons


if __name__ == '__main__':
    tokens = 'از قصه کودکیشان که می‌گفت، گاهی حرص می‌خورد'

    model = cons.load_model()
    resutls_cons = cons.cons_parser(model, tokens)
    resutls_chunks = cons.chunker(resutls_cons)
    print('Constituency Parses: \n', resutls_cons)
    print('Chunks: \n', resutls_chunks)
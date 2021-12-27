import dadmatools.models.postagger as pos


if __name__ == '__main__':
    tokens = [['از', 'قصهٔ', 'کودکی', 'شان' ,'که', 'می‌گفت', '،', 'گاهی', 'حرص', 'می‌خورد']]

    model = pos.load_model()
    resutls = pos.postagger(model, tokens)
    print(resutls)
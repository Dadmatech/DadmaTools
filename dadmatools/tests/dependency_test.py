import dadmatools.models.dependancy_parser as dep


if __name__ == '__main__':
    tokens = [['از', 'قصهٔ', 'کودکی', 'شان' ,'که', 'می‌گفت', '،', 'گاهی', 'حرص', 'می‌خورد']]

    model = dep.load_model()
    resutls = dep.depparser(model, tokens)
    print(resutls)
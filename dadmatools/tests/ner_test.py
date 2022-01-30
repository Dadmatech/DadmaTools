import dadmatools.models.ner as ner

if __name__ == '__main__':
    text = 'این سریال به صورت رسمی در تاریخ دهم می ۲۰۱۱ توسط علی مرادی برای پخش رزرو شد'

    model = ner.load_model()
    results = ner.ner(model, text)
    print(results)
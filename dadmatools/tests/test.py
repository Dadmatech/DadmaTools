from pipeline import language


nlp = language.Pipeline('lem')
print(nlp.pipe_names)
print(nlp.analyze_pipes(pretty=True))

doc = nlp('من دیروز به کتابخانه رفتم! فردا به مدرسه می‌روم.')
print(doc)
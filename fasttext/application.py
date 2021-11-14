from gensim.models import FastText

model = FastText.load('fasttext.model')

print(model.wv.similarity('你', '是'))  # 求相似
print(model.wv['你'])
print(model.wv.most_similar("滋润"))

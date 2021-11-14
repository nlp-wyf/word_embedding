from gensim.models import word2vec


model = word2vec.Word2Vec.load("word2vec.model")

# 输出某一个词对应的字向量表示
print(model.wv['傅小司'])

# 在所给的几个词中找出最不同的一个词
print('-' * 30)
print(model.wv.doesnt_match(u"傅小司 立夏 陆之昂 夏天".split()))

# 输出和所给词最相似的三个词
print('-' * 30)
req_count = 3
for key in model.wv.similar_by_word('傅小司', topn=100):
    if len(key[0]) > 1:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

# 查看两个词向量的相近程度
print('-' * 30)
print(model.wv.similarity('傅小司', '陆之昂'))
print(model.wv.similarity('可乐', '立夏'))

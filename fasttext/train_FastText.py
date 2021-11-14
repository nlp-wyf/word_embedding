import jieba
from gensim.models import FastText
from gensim.test.utils import get_tmpfile


def get_sentences(data_path):
    sentences = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_ls = jieba.lcut(line.strip())
            sentences.append(word_ls)
    return sentences


# 加载语料
data_path = 'corpus.zh'
sentences = get_sentences(data_path)

# 训练语料
model = FastText(sentences=sentences, vector_size=100, window=3, min_count=1, min_n=3, max_n=6)
model.save("fasttext.model")


fname = 'test_fasttext.txt'
model.wv.save_word2vec_format(fname, binary=False)

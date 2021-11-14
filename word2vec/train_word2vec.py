# word2vec训练词向量
from gensim.models import word2vec
from gensim.test.utils import get_tmpfile

# 加载语料
sentences = word2vec.LineSentence('./夏至未至_segment.txt')
# 训练语料
path = get_tmpfile("word2vec.model")  # 创建临时文件
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=50)
model.save("word2vec.model")

fname = 'test_wordvec.txt'
model.wv.save_word2vec_format(fname, binary=False)

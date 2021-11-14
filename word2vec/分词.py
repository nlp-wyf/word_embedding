import jieba.analyse


# 使用jieba工具进行中文分词
jieba.suggest_freq('傅小司', True)
jieba.suggest_freq('立夏', True)
jieba.suggest_freq('陆之昂', True)
jieba.suggest_freq('遇见', True)
jieba.suggest_freq('程七七', True)
jieba.suggest_freq('段桥', True)
jieba.suggest_freq('李嫣然', True)
jieba.suggest_freq('青田', True)
jieba.suggest_freq('颜末', True)
jieba.suggest_freq('伍凯', True)
jieba.suggest_freq('罗旭', True)
jieba.suggest_freq('杰迅', True)

with open('./夏至未至.txt', encoding='utf-8') as f:
    document = f.read()

    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    with open('./夏至未至_segment.txt', 'w', encoding="utf-8") as f2:
        f2.write(result)

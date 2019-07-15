import os, time, re
import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"] # 用來正常顯示中文標籤
plt.rcParams["axes.unicode_minus"] = False # 用來正常顯示負號
from adjustText import adjust_text


strProjectPath = os.path.dirname(__file__)
strRAWDataPath = os.path.join(strProjectPath, "01-RAWData")
strSenDataPath = os.path.join(strRAWDataPath, "all_sents.txt")
strOutputPath = os.path.join(strProjectPath, "Output/word2vec")
strModelPath = os.path.join(strProjectPath, "Model/word2vec")


jieba.set_dictionary(os.path.join(strRAWDataPath, "dict.txt.big.txt"))

Seg_Word_file_Path = "corpus_Seg.txt"
if not os.path.exists(os.path.join(strOutputPath, Seg_Word_file_Path)):

    # read original sentences, use jieba cut off and store in a list
    Train_Seg_list = []
    with open(strSenDataPath, "r", encoding="utf-8") as Sents:
        for sents in Sents:
            Train_Seg_list.append([" ".join(list(jieba.cut(sents, cut_all=False)))])
            print(sents)


    # save all seg from jieba list 
    with open(os.path.join(strOutputPath, Seg_Word_file_Path), "wb") as fW:
        for i in range(len(Train_Seg_list)):
            fW.write(Train_Seg_list[i][0].encode("utf-8"))


if not os.path.exists(os.path.join(strModelPath, "word2vec")):
    lines = []
    with open(os.path.join(strOutputPath, Seg_Word_file_Path), "r", encoding="utf-8") as f:
        for s in f:
            lines.append(s)

    print("Training word2vec...")
    w2v = Word2Vec(lines, size=300, min_count=20, iter=20)
    w2v.save(os.path.join(strModelPath, "word2vec"))
else:
    w2v = Word2Vec.load(os.path.join(strModelPath, "word2vec"))


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        if 6000 >= w2v.wv.vocab[word].count >= 3000:
            tokens.append(model[word])
            labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     textcoords="offset points",
                     xytext=(5, 2),
                     ha="right",
                     va="bottom")

    plt.savefig(os.path.join(strOutputPath, "visWord2vec.png"))

tsne_plot(model=w2v)

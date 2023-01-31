import pickle
import torch
import numpy as np
from src.evaluation.word_translation import get_word_translation_accuracy

# from Other_models.vecmap import embeddings

langx = 'en'
# langz = 'ru'
# tgt_dir = "en_ru"
dico_eval = "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/Dinu/"
# word2id_x_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_en_mid011/worddict.pkl"
# word2id_z_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_mid011/worddict.pkl"
# emb_x_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_en_mid011/embeddings.pkl"
# emb_z_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_mid011/embeddings.pkl"
# word2id_x_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_en_mid011/worddict.pkl"
# word2id_z_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_mid011/worddict.pkl"
# emb_x_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_en_mid011/embeddings.pkl"
# emb_z_file = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/XZW/ru_mid011/embeddings.pkl"


# with open(word2id_x_file, 'rb') as pkl_file:
#     word2id_x = pickle.load(pkl_file)
# with open(word2id_z_file, 'rb') as pkl_file:
#     word2id_z = pickle.load(pkl_file)
# with open(emb_x_file, 'rb') as pkl_file:
#     xw = pickle.load(pkl_file)
# with open(emb_z_file, 'rb') as pkl_file:
#     zw = pickle.load(pkl_file)
# print("load all files")

# for langz in ["ar","bg","de","es","fi","fr","hi","it","ja","tr","zh"]:
for langz in ["de", "es", "it", "fi"]:
    tgt_dir = "en_"+langz

    src_embeddings = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/"+tgt_dir+"/en.vec"
    trg_embeddings = "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/"+tgt_dir+"/"+langz+".vec"

    def read(file, threshold=0, vocabulary=None, dtype='float'):
        header = file.readline().split(' ')
        count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
        dim = int(header[1])
        words = []
        matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
        for i in range(count):
            word, vec = file.readline().split(' ', 1)
            if vocabulary is None:
                words.append(word)
                matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
            elif word in vocabulary:
                words.append(word)
                matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
        # matrix = torch.from_numpy(matrix).float()
        return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


    print("Read src embeddings from %s" % src_embeddings)
    srcfile = open(src_embeddings, encoding='utf-8', errors='surrogateescape')
    src_words, xw = read(srcfile, threshold=500000)
    print("Read trg embeddings from %s" % trg_embeddings)
    trgfile = open(trg_embeddings, encoding='utf-8', errors='surrogateescape')
    trg_words, zw = read(trgfile, threshold=500000)

    # xw=xw[:200000]
    # zw=zw[:200000]
    xw=torch.tensor(xw).cuda()
    zw=torch.tensor(zw).cuda()

    word2id_x = {word: i for i, word in enumerate(src_words)}
    word2id_z = {word: i for i, word in enumerate(trg_words)}

    print("Start to get translation accuracy...")
    get_word_translation_accuracy(
        langx, word2id_x, xw,
        langz, word2id_z, zw,
        method='csls_knn_10',
        dico_eval=dico_eval
    )

    # get_word_translation_accuracy(
    #     langz, word2id_z, zw,
    #     langx, word2id_x, xw,
    #     method='csls_knn_10',
    #     dico_eval=dico_eval
    # )
    del xw, zw
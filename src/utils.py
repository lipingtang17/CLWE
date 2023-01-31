# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE) and unsup (https://github.com/taasnim/unsup-word-translation)


import os
import io
import re
import sys
import pickle
import random
import inspect
import argparse
import subprocess
import fnmatch
import numpy as np
import torch
from torch import optim
from logging import getLogger

from .logger import create_logger
from .dictionary import Dictionary
from esim.data import Preprocessor, Preprocessor_XNLI

MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')
logger = getLogger()

# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
#     FAISS_AVAILABLE = True
    FAISS_AVAILABLE = False
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
#     if not params.exp_path:
    params.exp_path = get_exp_path(params)
    with io.open(os.path.join(params.exp_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0 
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        # index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.detach().numpy() if all_distances.requires_grad else all_distances.numpy()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            optim_params[split[0]] = float(split[1])
        print(optim_params)
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = MAIN_DUMP_PATH if params.exp_path == '' else params.exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    assert params.exp_name != ''
    exp_folder = os.path.join(exp_folder, params.exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        # assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def read_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    _emb_dim_file = params.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                #assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        logger.warning("Word '%s' found twice in %s embedding file"
                                       % (word, 'source' if source else 'target'))
                else:
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings
 
    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def select_subset(word_list, max_vocab):
    """
    Select a subset of words to consider, to deal with words having embeddings
    available in different casings. In particular, we select the embeddings of
    the most frequent words, that are usually of better quality.
    """
    word2id = {}
    indexes = []
    for i, word in enumerate(word_list):
        word = word.lower()
        if word not in word2id:
            word2id[word] = len(word2id)
            indexes.append(i)
        if max_vocab > 0 and len(word2id) >= max_vocab:
            break
    assert len(word2id) == len(indexes)
    return word2id, torch.LongTensor(indexes)


def load_pth_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a PyTorch binary file.
    """
    # reload PyTorch binary file
    lang = params.src_lang if source else params.tgt_lang
    data = torch.load(params.src_emb if source else params.tgt_emb)
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    dico = data['dico']
    embeddings = data['vectors']
    assert dico.lang == lang
    assert embeddings.size() == (len(dico), params.emb_dim)
    logger.info("Loaded %i pre-trained word embeddings." % len(dico))

    # select a subset of word embeddings (to deal with casing)
    if not full_vocab:
        word2id, indexes = select_subset([dico[i] for i in range(len(dico))], max_vocab)
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, lang)
        embeddings = embeddings[indexes]

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_bin_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = params.src_lang if source else params.tgt_lang
    model = load_fasttext_model(params.src_emb if source else params.tgt_emb)
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    words = model.get_labels()
    assert model.get_dimension() == params.emb_dim
    logger.info("Loaded binary model. Generating embeddings ...")
    embeddings = torch.from_numpy(np.concatenate([model.get_word_vector(w)[None] for w in words], 0))
    logger.info("Generated embeddings for %i words." % len(words))
    assert embeddings.size() == (len(words), params.emb_dim)

    # select a subset of word embeddings (to deal with casing)
    if not full_vocab:
        word2id, indexes = select_subset(words, max_vocab)
        embeddings = embeddings[indexes]
    else:
        word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_embeddings(params, source, full_vocab=False):
    """
    Reload pretrained embeddings.
    - `full_vocab == False` means that we load the `params.max_vocab` most frequent words.
      It is used at the beginning of the experiment.
      In that setting, if two words with a different casing occur, we lowercase both, and
      only consider the most frequent one. For instance, if "London" and "london" are in
      the embeddings file, we only consider the most frequent one, (in that case, probably
      London). This is done to deal with the lowercased dictionaries.
    - `full_vocab == True` means that we load the entire embedding text file,
      before we export the embeddings at the end of the experiment.
    """
    assert type(source) is bool and type(full_vocab) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    if emb_path.endswith('.pth'):
        return load_pth_embeddings(params, source, full_vocab)
    if emb_path.endswith('.bin'):
        return load_bin_embeddings(params, source, full_vocab)
    else:
        return read_txt_embeddings(params, source, full_vocab)


def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None


def export_embeddings(word2idz, zw, word2idx, xw,  langz, inputdir_mnli, inputdir_xnli,
                      targetdir_mnli_en, targetdir_xnli_en, targetdir_xnli_z):

    if not os.path.exists(targetdir_mnli_en):
        os.makedirs(targetdir_mnli_en)
    if not os.path.exists(targetdir_xnli_en):
        os.makedirs(targetdir_xnli_en)
    if not os.path.exists(targetdir_xnli_z):
        os.makedirs(targetdir_xnli_z)

    preprocessor_mnli = Preprocessor(mapping_dir=None,
                                     lowercase=False,
                                     ignore_punctuation=False,
                                     num_words=None,
                                     stopwords=[],
                                     labeldict={"entailment": 0, "neutral": 1, "contradiction": 2},
                                     bos="_BOS_",
                                     eos="_EOS_")
    preprocessor_xnli = Preprocessor_XNLI(language=langz,
                                        mapping_dir=None,
                                        lowercase=False,
                                        ignore_punctuation=False,
                                        num_words=None,
                                        stopwords=[],
                                        labeldict={"entailment": 0, "neutral": 1, "contradiction": 2},
                                        bos="_BOS_",
                                        eos="_EOS_")

    data_mnli_en = {}
    data_xnli_z = {}
    data_xnli_en = {}
    # Add the files in the directory of mnli
    print("\t* Reading data from %s" % inputdir_mnli)
    for file in os.listdir(inputdir_mnli):
        # print(file)
        # if fnmatch.fnmatch(file, "*.txt") and file != "README.txt":
        if fnmatch.fnmatch(file, "*_train.txt") and file != "README.txt":
            data_tmp = preprocessor_mnli.read_data(os.path.join(inputdir_mnli, file))
            for key, value in data_tmp.items():
                data_mnli_en[key] = value + data_mnli_en.get(key, [])
                # print("data_tmp-key: ", data_tmp.keys())
                # print("type(value: ", type(value))
                # print("len(data_temp-value): ", len(value))
            # data_mnli_en = dict(data_mnli_en, **data_tmp)

    # Retrieve the train, dev and test data files from the dataset directory.
    print("\t* Reading data from %s" % inputdir_xnli)
    for file in os.listdir(inputdir_xnli):
        if fnmatch.fnmatch(file, "*.tsv"):
            data_tmp = preprocessor_xnli.read_data(os.path.join(inputdir_xnli, file), "en")
            # data_xnli_en = dict(data_xnli_en, **data_tmp)
            for key, value in data_tmp.items():
                data_xnli_en[key] = value + data_xnli_en.get(key, [])
            data_tmp = preprocessor_xnli.read_data(os.path.join(inputdir_xnli, file), langz)
            # data_xnli_z = dict(data_xnli_z, **data_tmp)
            for key, value in data_tmp.items():
                data_xnli_z[key] = value + data_xnli_z.get(key, [])

    print("\t* Computing MNLI en worddict and saving it...")
    # data_en = {}
    # for key, value in data_xnli_en.items():
    #     data_en[key] = value + data_mnli_en.get(key, [])
    data_en = data_mnli_en
    preprocessor_mnli.build_worddict(data_en)
    with open(os.path.join(targetdir_mnli_en, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor_mnli.worddict, pkl_file)

    print("\t* Building MNLI en embedding matrix and saving it...")
    embed_matrix = build_embedding_matrix(xw, word2idx, preprocessor_mnli.worddict)
    with open(os.path.join(targetdir_mnli_en, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)

    print("\t* Computing XNLI en worddict and saving it...")
    preprocessor_xnli.build_worddict(data_xnli_en)
    with open(os.path.join(targetdir_xnli_en, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor_xnli.worddict, pkl_file)

    print("\t* Building XNLI en embedding matrix and saving it...")
    embed_matrix = build_embedding_matrix(xw, word2idx, preprocessor_xnli.worddict)
    with open(os.path.join(targetdir_xnli_en, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)

    print("\t* Computing XNLI %s worddict and saving it..." % langz)
    preprocessor_xnli.build_worddict(data_xnli_z)
    with open(os.path.join(targetdir_xnli_z, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor_xnli.worddict, pkl_file)

    print("\t* Building XNLI %s embedding matrix and saving it..." % langz)
    embed_matrix = build_embedding_matrix(zw, word2idz, preprocessor_xnli.worddict)
    with open(os.path.join(targetdir_xnli_z, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


def build_embedding_matrix(xw, word2idx, word2idx_nli):
    """
    Build an embedding matrix with pretrained weights for object's
    worddict.

    Args:
        embeddings_file: A file containing pretrained word embeddings.

    Returns:
        A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
        containing pretrained word embeddings (the +n_special_tokens is for
        the padding and out-of-vocabulary tokens, as well as BOS and EOS if
        they're used).
    """
    num_words = len(word2idx_nli)
    embedding_dim = xw.shape[1]
    embedding_matrix = np.zeros((num_words, embedding_dim))
    print("xw: ", xw)

    missed = 0
    for word, i in word2idx_nli.items():
        if word in word2idx:
            embedding_matrix[i] = np.array(xw[word2idx[word]].cpu(), dtype=float)
        else:
            if word == "_PAD_":
                continue
            missed += 1
            # Out of vocabulary words are initialised with random gaussian samples.
            embedding_matrix[i] = np.random.normal(size=(embedding_dim))
    print("\t* There are %d words in the training set" % num_words)
    print("\t* Missed words: ", missed)
    print("embedding_matrix: ", embedding_matrix)

    return embedding_matrix


def read_data_mnli(filepath):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    with open(filepath, "r", encoding="utf8") as input_data:
        ids, premises, hypotheses, labels = [], [], [], []

        # Translation tables to remove parentheses and punctuation from strings.
        parentheses_table = str.maketrans({"(": None, ")": None})

        # Ignore the headers on the first line of the file.
        next(input_data)

        for line in input_data:
            line = line.strip().split("\t")

            # Ignore sentences that have no gold label.
            if line[0] == "-":
                continue

            pair_id = line[7]
            premise = line[1]
            hypothesis = line[2]

            # Remove '(' and ')' from the premises and hypotheses.
            premise = premise.translate(parentheses_table)
            hypothesis = hypothesis.translate(parentheses_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append(premise.rstrip().split())
            hypotheses.append(hypothesis.rstrip().split())
            labels.append(line[0])
            ids.append(pair_id)

        print("\t* There are %d premises and %d hypotheses in %s" % (len(premises), len(hypotheses), filepath))

        return {"ids": ids,
                "premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


def read_data_xnli(filepath, lang):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    with open(filepath, "r", encoding="utf8") as input_data:
        ids, premises, hypotheses, labels = [], [], [], []

        # Ignore the headers on the first line of the file.
        next(input_data)

        for line in input_data:
            line = line.strip().split("\t")

            # Ignore sentences that are not in the target language
            if line[0] != lang:
                continue

            pair_id = line[9]
            premises.append(line[-3].split())
            hypotheses.append(line[-2].split())
            labels.append(line[1])
            ids.append(pair_id)

        print("\t* There are %d premises and %d hypotheses in %s" % (len(premises), len(hypotheses), filepath))

        return {"ids": ids,
                "premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}

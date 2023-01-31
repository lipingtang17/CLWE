# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE) and unsup (https://github.com/taasnim/unsup-word-translation)


import os
import io
from logging import getLogger
import numpy as np
import torch
from torch import Tensor as torch_tensor

from ..utils import get_nn_avg_dist



logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2 and len(w1) > 1]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
#     if dico_eval == 'default':
#         path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
#     elif dico_eval == 'vecmap':
#         path = os.path.join(DIC_EVAL_PATH, 'vecmap/%s-%s.5000-6500.txt' % (lang1, lang2))
#     else:
#         path = dico_eval
    if type(emb1) == np.ndarray:
        emb1 = torch.from_numpy(emb1)
        emb2 = torch.from_numpy(emb2)

    if "Dinu" in dico_eval:
        path = os.path.join(dico_eval, '%s-%s.test.txt' % (lang1, lang2))
    else:
        path = os.path.join(dico_eval, '%s-%s.5000-6500.txt' % (lang1, lang2))
        # path = os.path.join(dico_eval, '%s-%s.txt' % (lang1, lang2))
    dico = load_dictionary(path, word2id1, word2id2)  # Return a torch tensor of size (n, 2) containing word2id indexes
    dico = dico.cuda() if emb1.is_cuda else dico

    # logger.info("Evaluate on the dictionary: %s" % path)
    print("Evaluate on the dictionary: %s" % path)

    # assert dico[:, 0].max() < emb1.size(0)
    # assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    
    # emb1 = emb1[:200000]
    # emb2 = emb2[:200000]
    
    batch_size = 3000
    orig_dico = dico
    top_matches = None
    for i in range(0, len(orig_dico), batch_size):
        dico = orig_dico[i:i+batch_size,:]

        # nearest neighbors
        if method == 'nn':
            query = emb1[dico[:, 0]]
            scores = query.mm(emb2.transpose(0, 1))

        # inverted softmax
        elif method.startswith('invsm_beta_'):
            beta = float(method[len('invsm_beta_'):])
            bs = 128
            word_scores = []
            for i in range(0, emb2.size(0), bs):
                scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
                scores.mul_(beta).exp_()
                scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
                word_scores.append(scores.index_select(0, dico[:, 0]))
            scores = torch.cat(word_scores, 1)

        # contextual dissimilarity measure
        elif method.startswith('csls_knn_'):
            # average distances to k nearest neighbors
            knn = method[len('csls_knn_'):]
            assert knn.isdigit()
            knn = int(knn)
            # import ipdb
            # ipdb.set_trace()
            average_dist1 = get_nn_avg_dist(emb2, emb1, knn) 
            average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
            average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
            average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
            # queries / scores
            query = emb1[dico[:, 0]]
            scores = query.mm(emb2.transpose(0, 1))
            scores.mul_(2)
            scores.sub_(average_dist1[dico[:, 0]][:, None])
            scores.sub_(average_dist2[None, :])
        elif method.startswith('nn'):
            query = emb1[dico[:, 0]]
            scores = query.mm(emb2.transpose(0, 1))
        else:
            raise Exception('Unknown method: "%s"' % method)

        # results = []
        # import ipdb
        # ipdb.set_trace()
        if top_matches != None:
            top_matches = torch.cat((top_matches, scores.topk(10, 1, True)[1]), 0)
        else:
            top_matches = scores.topk(10, 1, True)[1]
        # for k in [1, 5, 10]:
        #     top_k_matches = top_matches[:, :k]
        #     _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        #     # allow for multiple possible translations
        #     matching = {}
        #     for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
        #         matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        #     # # evaluate precision@k
        #     precision_at_k = 100 * np.mean(torch.Tensor(list(matching.values())).numpy())
        #     logger.info("%i source words - %s - Precision at k = %i: %f" %
        #                 (len(matching), method, k, precision_at_k))
    
    # import ipdb; ipdb.set_trace()
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == orig_dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(orig_dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # # evaluate precision@k
        precision_at_k = 100 * np.mean(torch.Tensor(list(matching.values())).numpy())
        # logger.info("%i source words - %s - Precision at k = %i: %f" %
                    # (len(matching), method, k, precision_at_k))
        print("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))


def dict_mean_cosine(params, dict_path, src_emb, tgt_emb, src_to_tgt):
    """
    Mean-cosine model selection criterion.
    """

    if src_to_tgt:
        # get normalized embeddings

        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        dico = load_dictionary(dict_path, params.src_dico.word2id,
                               params.tgt_dico.word2id)  # Return a torch tensor of size (n, 2) containing word2id indexes
        dico = dico.cuda() if src_emb.is_cuda else dico

        mean_cosine = (src_emb[dico[:, 0]] * tgt_emb[dico[:, 1]]).sum(1).mean()
        mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
        logger.info("Mean cosine A->B: %.5f" % (mean_cosine))
        mean_dist = torch.sqrt(torch.sum((src_emb[dico[:, 0]]-tgt_emb[dico[:, 1]])**2, dim=1)).mean()
        logger.info("Mean distance A->B: %.5f" % (mean_dist))

    else:

        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        dico = load_dictionary(dict_path, params.tgt_dico.word2id, params.src_dico.word2id)  # Return a torch tensor of size (n, 2) containing word2id indexes
        dico = dico.cuda() if tgt_emb.is_cuda else dico

        mean_cosine = (tgt_emb[dico[:, 0]] * src_emb[dico[:, 1]]).sum(1).mean()
        mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
        logger.info("Mean cosine B->A: %.5f" % (mean_cosine))
        mean_dist = torch.sqrt(torch.sum((tgt_emb[dico[:, 0]] - src_emb[dico[:, 1]]) ** 2, dim=1)).mean()
        logger.info("Mean distance B->A: %.5f" % (mean_dist))

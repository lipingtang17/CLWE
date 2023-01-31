# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE) and unsup (https://github.com/taasnim/unsup-word-translation)

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm

from src.utils import bool_flag, initialize_exp, export_embeddings
from src.models import build_model, init_weight_linear
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.refinement import generate_new_dictionary, symmetric_reweighting
from src.evaluation.word_translation import get_word_translation_accuracy, dict_mean_cosine, load_dictionary
from torch.distributions.beta import Beta


VALIDATION_METRIC_AB = 'mean_cosine-csls_knn_10-S2T-10000'
VALIDATION_METRIC_BA = 'mean_cosine-csls_knn_10-T2S-10000'

# VALIDATION_METRIC_AB = 'number_words_in_dict-S2T'
# VALIDATION_METRIC_BA = 'number_words_in_dict-T2S'

# VALIDATION_METRIC_AB = 'number_words_in_dict*cos-S2T'
# VALIDATION_METRIC_BA = 'number_words_in_dict*cos-T2S'

# main
parser = argparse.ArgumentParser(description='Unsupervised robust training using domain flow interpolation')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="mid", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='de', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--emb_dim_autoenc", type=int, default=350, help="Embedding dimension in the latent space")
parser.add_argument("--max_vocab_A", type=int, default=200000, help="Maximum vocabulary size for source (-1 to disable)")
parser.add_argument("--max_vocab_B", type=int, default=200000, help="Maximum vocabulary size for target (-1 to disable)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# mapping
# parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_init", type=str, default="identity", choices=["identity", "id_char", "supervised"], help="Initialize the mapping")
parser.add_argument("--dico_trainA", type=str, default="", help="The dictionary used fro supervised training")
parser.add_argument("--dico_trainB", type=str, default="", help="The dictionary used fro supervised training")
parser.add_argument("--map_beta", type=float, default=0.01, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--cycle_lambda", type=float, default=5, help="Cycle loss feedback coefficient")
parser.add_argument("--reconstruction_lambda", type=float, default=1, help="Reconstruction loss feedback coefficient")
parser.add_argument("--dis_most_frequent_AB", type=int, default=20000, help="Select embeddings of the k most frequent words for discrimination in source to target (0 to disable)")
parser.add_argument("--dis_most_frequent_BA", type=int, default=20000, help="Select embeddings of the k most frequent words for discrimination in target to source (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.2, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=100000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="adam,lr=0.001", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.001", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_procrustes", type=int, default=100, help="Maximum number of procurstes iterations in refinement procedure")
parser.add_argument("--n_symmetric_reweighting", type=int, default=50, help="Maximum number of symmetric reweighting iterations in refinement procedure")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="renorm,center,renorm", help="Normalize embeddings before training")
# Autoencoder
parser.add_argument("--autoenc_epochs", type=int, default=25, help="Number of initail autoencoder epochs")
parser.add_argument("--autoenc_optimizer", type=str, default="adam,lr=0.001", help="Autoencoder optimizer")
parser.add_argument("--out_tanh", type=int, default=0, help="Whether output layer of autoenconder use tanh activation")
parser.add_argument("--l_relu", type=int, default=0, help="Whether bottleneck of autoenconder use leaky relu activation")
# domain flow interpolation
parser.add_argument("--mid_domain", type=bool_flag, default=True, help="Whether to use mid domain adaptation")
parser.add_argument("--finetune_epochs", type=int, default=5, help="number of epochs to fine-tune with z=1")

parser.add_argument("--export_emb", type=bool_flag, default=False, help="Whether to export the generated embeddings")
parser.add_argument("--save_proc", type=bool_flag, default=False, help="Whether to save the procrusted mapping")

# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert os.path.isdir(params.dico_eval)


# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A, encoder_B, decoder_B \
    = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A,
                  decoder_A, encoder_B, decoder_B, params)
evaluator = Evaluator(trainer)


"""
Learning loop for Adversarial Training
"""
if params.adversarial:

    # first train the autoencoder to become mature
    trainer.train_autoencoder_A()
    trainer.train_autoencoder_B()

    if params.mid_domain:
        trainer.mapping_G.init_weight(params, params.dico_trainA, params.src_dico, params.tgt_dico, src_emb, tgt_emb, trainer.encoder_A, trainer.encoder_B)
        trainer.mapping_F.init_weight(params, params.dico_trainB, params.tgt_dico, params.src_dico, tgt_emb, src_emb, trainer.encoder_B, trainer.encoder_A)
    else:
        init_weight_linear(mapping_G, params, params.dico_trainA, params.src_dico, params.tgt_dico, src_emb, tgt_emb, trainer.encoder_A, trainer.encoder_B)
        init_weight_linear(mapping_F, params, params.dico_trainB, params.tgt_dico, params.src_dico, tgt_emb, src_emb, trainer.encoder_B, trainer.encoder_A)

    logger.info('----> ADVERSARIAL TRAINING <----\n\n')

    # adversarial training loop
    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc_G = 0
        n_words_proc_F = 0
        stats = {'DIS_COSTS_A': [], 'DIS_COSTS_B': [], 'z': 0} if params.mid_domain else {'DIS_COSTS_A': [], 'DIS_COSTS_B': []}

        # set the distribution of domain variable z
        if params.mid_domain:
            t = n_epoch
            T = params.n_epochs - params.finetune_epochs
            alpha = torch.exp(torch.tensor((t-0.5*T)/(0.25*T)))
            m = Beta(alpha, torch.tensor([1.0]))

        for n_iter in tqdm(range(0, params.epoch_size, params.batch_size)):
            # sample z from m
            if params.mid_domain:
                z = m.sample().cuda() if n_epoch < params.n_epochs-params.finetune_epochs else torch.tensor([1.0]).type_as(trainer.tgt_emb.weight.data).cuda()
                stats['z'] = z
        
            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step_B(stats)
                trainer.dis_step_A(stats)

            # mapping training 
            n_words_proc_G += trainer.mapping_step_G(stats)
            n_words_proc_F += trainer.mapping_step_F(stats)

            # log stats
            if n_iter % 1000 == 0:
                stats_str = [('DIS_COSTS_B', 'Discriminator B loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc_G / (time.time() - tic)))
                
                n_words_proc_G = 0
                for k, _ in stats_str:
                    del stats[k][:]

                stats_str = [('DIS_COSTS_A', 'Discriminator A loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc_F / (time.time() - tic)))
                # reset
                tic = time.time()
                n_words_proc_F = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # model evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        evaluator.model_selection_criterion(to_log)
        
        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best_AB(to_log, VALIDATION_METRIC_AB)
        trainer.save_best_BA(to_log, VALIDATION_METRIC_BA)
        logger.info('End of epoch %i.\n\n' % n_epoch)

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC_AB, VALIDATION_METRIC_BA)
        if trainer.map_optimizer_G.param_groups[0]['lr'] < params.min_lr or trainer.map_optimizer_F.param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break


"""
Learning loop for Refinement Procedure
"""

# Refinement for language A->B
print("\n \n Refinement iteration for ", params.src_lang, "to ", params.tgt_lang,  "\n")

# Reload best model from adversarial training
if params.n_epochs > 0 or params.map_init!="id_char":
    trainer.reload_best_AB()
trainer.build_dictionary_AB()

prev_score_mean = 0

# apply the Procrustes solution for language A->B
print('**Procrustes solution iteration for  ', params.src_lang, "to ", params.tgt_lang,  ' **')
for n_iter in range(params.n_procrustes):
    trainer.procrustes_AB()

    # build a dictionary from aligned embeddings
    if params.mid_domain:
        emb1 = (trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data), 1).data)[0:params.dico_max_rank]
    else:
        emb1 = (trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data)).data)[0:params.dico_max_rank]
    emb2 = (trainer.encoder_B(trainer.tgt_emb.weight.data).data)[0:params.dico_max_rank]
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    all_pairs, all_scores = generate_new_dictionary(emb1, emb2)

    trainer.dico_AB = all_pairs.cuda() if params.cuda else all_pairs

    score_mean = all_scores[:, 0].mean().item()

    # checking threshold
    if score_mean - prev_score_mean >= 1e-06:
        prev_score_mean = score_mean
    elif n_iter > 20:
        break

logger.info('Finished %i procrustes iteration ... for A to B' % n_iter)
to_log = OrderedDict({'n_iter': n_iter})

get_word_translation_accuracy(
    params.src_dico.lang, params.src_dico.word2id,
    trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data),1).data if params.mid_domain
    else trainer.mapping_G(trainer.encoder_A(trainer.src_emb.weight.data)).data,
    params.tgt_dico.lang, params.tgt_dico.word2id, (trainer.encoder_B(trainer.tgt_emb.weight.data)).data,
    method='csls_knn_10',
    dico_eval=params.dico_eval
)

# Symmetric Reweighting for language A->B
print("** Symmetric Reweighting for ", params.src_lang, "to ", params.tgt_lang,  " **")
for i in range(params.n_symmetric_reweighting):
    # seed dictionary from previous step
    src_indices = trainer.dico_AB.cpu().numpy()[:,0]
    trg_indices = trainer.dico_AB.cpu().numpy()[:,1]
    xw, zw = symmetric_reweighting(src_emb, tgt_emb, src_indices, trg_indices)
    
    emb1 = xw[0:params.dico_max_rank]
    emb2 = zw[0:params.dico_max_rank]
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    all_pairs, all_scores = generate_new_dictionary(emb1, emb2)
    trainer.dico_AB = all_pairs.cuda() if params.cuda else all_pairs

    score_mean = all_scores[:, 0].mean().item()
    if score_mean - prev_score_mean >= 1e-06:
        prev_score_mean = score_mean
    elif i > 10:
        break 

logger.info('Finished %i Symmetric Re-weighting iteration ... for A to B' % i)
get_word_translation_accuracy(
        params.src_dico.lang, params.src_dico.word2id, xw,
        params.tgt_dico.lang, params.tgt_dico.word2id, zw,
        method='csls_knn_10',
        dico_eval=params.dico_eval
    )

if "Dinu" not in params.dico_eval:
    # Refinement for language B->A
    print("\n \n Refinement iteration for ", params.tgt_lang, "to ", params.src_lang,  "\n")

    # Reload best model from adversarial training
    if params.n_epochs > 0 or params.map_init!="id_char":
        trainer.reload_best_BA()
    trainer.build_dictionary_BA()

    prev_score_mean = 0
    # apply the Procrustes solution for language B->A
    print('** Procrustes solution iteration for  ', params.tgt_lang, "to ", params.src_lang,  ' **')
    for n_iter in range(params.n_procrustes):
        trainer.procrustes_BA()

        # build a dictionary from aligned embeddings
        emb1 = (trainer.encoder_A(trainer.src_emb.weight.data)).data[0:params.dico_max_rank]
        if params.mid_domain:
            emb2 = trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data),1).data[0:params.dico_max_rank]
        else:
            emb2 = trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data)).data[0:params.dico_max_rank]
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
        all_pairs, all_scores = generate_new_dictionary(emb2, emb1)

        trainer.dico_BA = all_pairs.cuda() if params.cuda else all_pairs

        score_mean = all_scores[:, 0].mean().item()
        # checking threshold
        if score_mean - prev_score_mean >= 1e-06:
            prev_score_mean = score_mean
        elif n_iter > 20:
            break

    logger.info('Finished %i procrustes iteration ... for B to A' % n_iter)
    to_log = OrderedDict({'n_iter': n_iter})

    # save the mapping
    # saving weight matrix of F
    if params.save_proc:
        path = os.path.join(params.exp_path, 'best_mapping_BA_proc.pth')
        if params.mid_domain:
            W = mapping_F.generator.weight.data.cpu().numpy()
        else:
            W = mapping_F.weight.data.cpu().numpy()
        logger.info('* Saving the mapping to %s ...' % path)
        torch.save(W, path)

    get_word_translation_accuracy(
        params.tgt_dico.lang, params.tgt_dico.word2id,
        trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data),1).data if params.mid_domain
        else trainer.mapping_F(trainer.encoder_B(trainer.tgt_emb.weight.data)).data,
        params.src_dico.lang, params.src_dico.word2id, (trainer.encoder_A(trainer.src_emb.weight.data)).data,
        method='csls_knn_10',
        dico_eval=params.dico_eval
    )

    # Symmetric Reweighting for language B->A
    print("** Symmetric Reweighting for ", params.tgt_lang, "to ", params.src_lang,  " **")
    i = 0
    for i in range(params.n_symmetric_reweighting):
        # seed dictionary from previous step
        src_indices = trainer.dico_BA.cpu().numpy()[:,0]
        trg_indices = trainer.dico_BA.cpu().numpy()[:,1]
        zw, xw = symmetric_reweighting(tgt_emb, src_emb, src_indices, trg_indices)

        emb1 = xw[0:params.dico_max_rank]
        emb2 = zw[0:params.dico_max_rank]
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
        all_pairs, all_scores = generate_new_dictionary(emb2, emb1)
        trainer.dico_BA = all_pairs.cuda() if params.cuda else all_pairs

        score_mean = all_scores[:, 0].mean().item()
        if score_mean - prev_score_mean >= 1e-06:
            prev_score_mean = score_mean
        elif i > 10:
            break

    logger.info('Finished %i Symmetric Re-weighting iteration ... for B to A' % i)
    get_word_translation_accuracy(
            params.tgt_dico.lang, params.tgt_dico.word2id, zw,
            params.src_dico.lang, params.src_dico.word2id, xw,
            method='csls_knn_10',
            dico_eval=params.dico_eval
        )
    print("xw embeddings shape: ", xw.shape)
    print("zw embeddings shape: ", zw.shape)

if params.export_emb:
    export_embeddings(params.tgt_dico.word2id, zw/zw.norm(2, 1, keepdim=True).expand_as(zw),
                      params.src_dico.word2id, xw/xw.norm(2, 1, keepdim=True).expand_as(xw),
                      params.tgt_dico.lang,
                      inputdir_mnli="/mnt/ssd-201-112-01/cpii.local/lptang/EvalDataset/multinli_1.0",
                      inputdir_xnli="/mnt/ssd-201-112-01/cpii.local/lptang/EvalDataset/XNLI-1.0",
                      targetdir_mnli_en="/users/cpii.local/lptang/NLP_models/CLWE_formal/Other_models/ESIM/data/preprocessed/MNLI/XZW/"
                                        + params.tgt_dico.lang + "_" + params.exp_name.split("/")[0] + params.exp_id + "_en",
                      targetdir_xnli_en="/users/cpii.local/lptang/NLP_models/CLWE_formal/Other_models/ESIM/data/preprocessed/XNLI/XZW/"
                                        + params.tgt_dico.lang + "_" + params.exp_name.split("/")[0] + params.exp_id + "_en",
                      targetdir_xnli_z="/users/cpii.local/lptang/NLP_models/CLWE_formal/Other_models/ESIM/data/preprocessed/XNLI/XZW/"
                                       + params.tgt_dico.lang + "_" + params.exp_name.split("/")[0] + params.exp_id
                      )
# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE) and unsup (https://github.com/taasnim/unsup-word-translation)

import scipy
import torch
from torch import nn
import torch.nn.functional as F

from .utils import load_embeddings, normalize_embeddings
from .evaluation.word_translation import load_identical_char_dico, load_dictionary


class Discriminator(nn.Module):

    def __init__(self, params, source=True):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim_autoenc
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)  

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


class Encoder(nn.Module):
    def __init__(self, params, source=True):
        super(Encoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.bottleneck_dim = params.emb_dim_autoenc
        self.l_relu = params.l_relu

        self.encoder = nn.Linear(self.emb_dim, self.bottleneck_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.encoder(x)
        if self.l_relu == 1:
            x = self.leakyRelu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, params, source=True):
        super(Decoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.bottleneck_dim = params.emb_dim_autoenc

        self.decoder = nn.Linear(self.bottleneck_dim, self.emb_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.decoder(x)
        return x


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.in_dim = params.emb_dim_autoenc
        self.out_dim = params.emb_dim_autoenc
        self.generator = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def init_weight(self, params, dico_train, src_dico, tgt_dico, src_emb, tgt_emb, encoder_A, encoder_B):
        if params.map_init == "identity":
            self.generator.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
        else:
            word2id1 = src_dico.word2id
            word2id2 = tgt_dico.word2id
            if params.map_init == "id_char":
                dico = load_identical_char_dico(word2id1, word2id2)
            elif params.map_init == "supervised":
                dico = load_dictionary(dico_train, word2id1, word2id2)
            self.procrustes(src_emb, tgt_emb, encoder_A, encoder_B, dico)

    def forward(self, x, z):
        mapped_x = self.generator(x)
        mid_domain = z*mapped_x + (1-z)*x
        return mid_domain

    def procrustes(self, src_emb, tgt_emb, encoder_A, encoder_B, dico):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = encoder_A(src_emb.weight.data[dico[:, 0]])
        B = encoder_B(tgt_emb.weight.data[dico[:, 1]])
        W = self.generator.weight.data
        M = B.transpose(0, 1).mm(A).cpu().detach().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


def init_weight_linear(model, params, dico_train, src_dico, tgt_dico, src_emb, tgt_emb, encoder_A, encoder_B):
    if params.map_init == "identity":
        model.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
    else:
        word2id1 = src_dico.word2id
        word2id2 = tgt_dico.word2id
        if params.map_init == "id_char":
            dico = load_identical_char_dico(word2id1, word2id2)
        elif params.map_init == "supervised":
            dico = load_dictionary(dico_train, word2id1, word2id2)
        procrustes_linear(model, src_emb, tgt_emb, encoder_A, encoder_B, dico)


def procrustes_linear(model, src_emb, tgt_emb, encoder_A, encoder_B, dico):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = encoder_A(src_emb.weight.data[dico[:, 0]])
        B = encoder_B(tgt_emb.weight.data[dico[:, 1]])
        W = model.weight.data
        M = B.transpose(0, 1).mm(A).cpu().detach().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    tgt_dico, _tgt_emb = load_embeddings(params, source=False)
    params.tgt_dico = tgt_dico
    tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
    tgt_emb.weight.data.copy_(_tgt_emb)


    # mapping
    if params.mid_domain:
        mapping_G = Generator(params)
        mapping_F = Generator(params)
        # mapping_G.init_weight(params, src_dico, tgt_dico, src_emb, tgt_emb)
        # mapping_F.init_weight(params, src_dico, tgt_dico, src_emb, tgt_emb)
    else:
        mapping_G = nn.Linear(params.emb_dim_autoenc, params.emb_dim_autoenc, bias=False)
        mapping_F = nn.Linear(params.emb_dim_autoenc, params.emb_dim_autoenc, bias=False)
        # if params.map_init == "identity":
        #     mapping_G.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
        #     mapping_F.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
    
    # discriminator
    discriminator_A = Discriminator(params) if with_dis else None
    discriminator_B = Discriminator(params) if with_dis else None

    # autoencoder
    encoder_A = Encoder(params)
    decoder_A = Decoder(params)
    encoder_B = Encoder(params)
    decoder_B = Decoder(params)

    # cuda
    if params.cuda:
        src_emb.cuda()
        tgt_emb.cuda()
        mapping_G.cuda()
        mapping_F.cuda()
        if with_dis:
            discriminator_A.cuda()
            discriminator_B.cuda()
        encoder_A.cuda()
        decoder_A.cuda()
        encoder_B.cuda()
        decoder_B.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)
    
    return src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A, encoder_B, decoder_B

"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import os
import string
import torch
import numpy as np

from collections import Counter
from torch.utils.data import Dataset
from torch import nn
from torch.autograd import Variable


def normalize_embeddings(emb, fix_idx, types="renorm,center,renorm", mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb[fix_idx, :].mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb) + 1e-13)
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
        # print("emb after ", t)
        # print(emb)
    return mean.cpu() if mean is not None else None


class Encoder(nn.Module):
    def __init__(self, bottleneck_dim):
        super(Encoder, self).__init__()

        self.emb_dim = 300
        self.bottleneck_dim = bottleneck_dim
        self.l_relu = 0

        self.encoder = nn.Linear(self.emb_dim, self.bottleneck_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.encoder(x)
        if self.l_relu == 1:
            x = self.leakyRelu(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.in_dim = 350
        self.out_dim = 350
        self.generator = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def forward(self, x, z):
        mapped_x = self.generator(x)
        mid_domain = z*mapped_x + (1-z)*x
        return mid_domain


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 mapping_dir=None,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.mapping_dir = mapping_dir
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def read_data(self, filepath):
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

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

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

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(line[0])
                ids.append(pair_id)

            print("\t* There are %d premises and %d hypotheses in %s" % (len(premises), len(hypotheses), filepath))

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)
        # print("words: ", words)
        # print("counts: ", counts)
        print("\t* There are %d words in the data" % num_words)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "labels": []}

        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
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
        # Load the word embeddings in a dictionnary.
        print("\t* Read original embeddings from %s" % embeddings_file)
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    # if word in self.worddict:
                    if word in self.worddict and len(line) > 2:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue
        # print("embeddings before processing: ", embeddings)

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        fix_idx = []
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
                fix_idx.append(i)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("\t* There are %d words in the training set" % num_words)
        print("\t* Missed words: ", missed)
        # import ipdb
        # ipdb.set_trace()
        
        # print("embedding_matrix before normalizing: ", embedding_matrix)
        if not self.mapping_dir:
            print(embedding_matrix)
            return embedding_matrix

        # print("embedding_matrix before normalizing: ", embedding_matrix)
        # print("embedding_matrix.shape: ", embedding_matrix.shape)
        # normalize embeddings
        embedding_matrix = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
        # print("embedding_matrix.shape: ", embedding_matrix.shape)
        normalize_embeddings(embedding_matrix, fix_idx)
        # print("embedding_matrix after normalizing: ", embedding_matrix)

        # load the encoder (and generator)
        if "MUSE" in self.mapping_dir:
            if "en" not in embeddings_file:
                print(embedding_matrix)
                return embedding_matrix
            else:
                mapping_path = os.path.join(self.mapping_dir, 'best_mapping.pth')
                print("Load mapping parameters from %s" % mapping_path)
                to_reload = torch.from_numpy(torch.load(mapping_path))
                mapping = nn.Linear(300, 300, bias=False)
                mapping.weight.data = to_reload
                print("\t* Map source embeddings to the target space ...")
                new_embedding_matrix = mapping(embedding_matrix).data
                return new_embedding_matrix
        else:
            # import ipdb
            # ipdb.set_trace()
            if "en" in embeddings_file:
                encoder_path = os.path.join(self.mapping_dir, 'best_encX_BA.pth')
            else:
                encoder_path = os.path.join(self.mapping_dir, 'best_encY_BA.pth')
                generator_path = os.path.join(self.mapping_dir, 'best_mapping_BA_proc.pth')
                # generator_path = os.path.join(self.mapping_dir, 'best_mapping_BA.pth')
                to_reload = torch.from_numpy(torch.load(generator_path))
                if "unsup_orig" in self.mapping_dir:
                    generator = nn.Linear(200, 200, bias=False)
                    # generator = nn.Linear(350, 350, bias=False)
                    generator.weight.data = to_reload 
                elif "orig" in self.mapping_dir:
                    generator = nn.Linear(350, 350, bias=False)
                    generator.weight.data = to_reload
                elif "mid" in self.mapping_dir:
                    generator = Generator()
                    # print(generator.generator.weight)
                    generator.generator.weight.data = to_reload
                print("\t* Load mapping parameters from %s" % generator_path)  
            if "unsup_orig" in self.mapping_dir:
                encoder = Encoder(200)
                # encoder = Encoder(350)
                to_reload = torch.from_numpy(torch.load(encoder_path))
                encoder.encoder.weight.data = to_reload  
            else:
                encoder = Encoder(350)
                encoder.load_state_dict(torch.load(encoder_path))
            print("\t* Load encoder parameters from %s" % encoder_path)

            # map source embeddings to the target space
            if "en" in embeddings_file:
                print("\t* Encode en embeddings using encoder from %s" % encoder_path)
                new_embedding_matrix = encoder(embedding_matrix).data
            else:
                print("\t* Map source embeddings to the target space ...")
                if "orig" in self.mapping_dir:
                    new_embedding_matrix = generator(encoder(embedding_matrix)).data
                elif "mid" in self.mapping_dir:
                    new_embedding_matrix = generator(encoder(embedding_matrix), 1).data
            print(new_embedding_matrix)

        return new_embedding_matrix



class Preprocessor_XNLI(Preprocessor):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 language="en",
                 mapping_dir=None,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        super(Preprocessor_XNLI, self).__init__()
        self.language = language
        self.mapping_dir = mapping_dir
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def read_data(self, filepath, language="en"):
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

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                # Ignore sentences that are not in the target language
                if line[0] != language:
                    continue

                pair_id = line[9]

                premises.append(line[-3].lower().split())
                hypotheses.append(line[-2].lower().split())
                labels.append(line[1])
                ids.append(pair_id)
            print("\t* There are %d premises and %d hypotheses in %s" % (len(premises), len(hypotheses), filepath))

            print("\t* premises example: ", premises[-1])
            print("\t* hypothesis example: ", hypotheses[-1])
            print("\t* labels exmaple: ", labels[-1])
            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}

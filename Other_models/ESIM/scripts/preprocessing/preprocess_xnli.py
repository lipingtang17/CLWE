"""
Preprocess the SNLI dataset and word embeddings to be used by the ESIM model.
"""
# Aurelien Coet, 2018.

import os
import pickle
import argparse
import fnmatch
import json

from esim.data import Preprocessor_XNLI


def preprocess_XNLI_data(inputdir,
                         embeddings_file,
                         targetdir,
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
    Preprocess the data from the SNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the dev and test data files from the dataset directory.
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*dev.tsv"):
            dev_file = file
        elif fnmatch.fnmatch(file, "*test.tsv"):
            test_file = file

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor_XNLI(mapping_dir=mapping_dir,
                                     language=language,
                                     lowercase=lowercase,
                                     ignore_punctuation=ignore_punctuation,
                                     num_words=num_words,
                                     stopwords=stopwords,
                                     labeldict=labeldict,
                                     bos=bos,
                                     eos=eos)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Reading dev and test set ", 20*"=")
    print("\t* Reading dev data...")
    data_dev = preprocessor.read_data(os.path.join(inputdir, dev_file), language)

    print("\t* Reading test data...")
    data_test = preprocessor.read_data(os.path.join(inputdir, test_file), language)

    if not os.path.exists(os.path.join(targetdir, "worddict.pkl")):
        print(20*"=", " Computing worddict and saving it ", 20*"=")
        data = {}
        for key, value in data_dev.items():
            data[key] = value + data_test.get(key, [])
        preprocessor.build_worddict(data)
        with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
            pickle.dump(preprocessor.worddict, pkl_file)
    else:
        print("\t* Loading worddict from %s..." % str(os.path.join(targetdir, "worddict.pkl")))
        with open(os.path.join(targetdir, "worddict.pkl"), "rb") as pkl_file:
            preprocessor.worddict = pickle.load(pkl_file)

    print(20*"=", " Transforming words in premises and hypotheses to indices", 20*"=")
    transformed_data = preprocessor.transform_to_indices(data_dev)
    print("\t* Saving (dev) result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    # print(20*"=", " Preprocessing test set ", 20*"=")
    # print("\t* Reading data...")
    # data = preprocessor.read_data(os.path.join(inputdir, test_file))

    # print("\t* Transforming words in premises and hypotheses (test) to indices...")
    transformed_data = preprocessor.transform_to_indices(data_test)
    print("\t* Saving (test) result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    if not os.path.exists(os.path.join(targetdir, "embeddings.pkl")):
        print("\t* Building embedding matrix and saving it...")
        embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
        with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
            pickle.dump(embed_matrix, pkl_file)


if __name__ == "__main__":
    default_config = "../../config/preprocessing/xnli_preprocessing/en_preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the XNLI dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing XNLI"
    )
    parser.add_argument("--mapping_dir",
                        default=None,
                        help="Path to a configuration file for preprocessing MultiNLI")
    parser.add_argument("--target_dir",
                        default="../../data/preprocessed/MNLI/zh_en",
                        help="Path to a configuration file for preprocessing MultiNLI")
    parser.add_argument("--lang",
                        default="en",
                        help="Path to a configuration file for preprocessing MultiNLI")
    parser.add_argument("--embeddings_file",
                        default="/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.zh.vec",
                        help="Path to a configuration file for preprocessing MultiNLI")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    preprocess_XNLI_data(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, args.embeddings_file)),
        os.path.normpath(os.path.join(script_dir, args.target_dir)),
        mapping_dir=args.mapping_dir,
        language=args.lang,
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        stopwords=config["stopwords"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"]
    )

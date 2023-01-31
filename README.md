# Unsupervised and Robust Cross-Lingual Word Embedding using Domain Flow Interpolation.

This repository contains the code of our paper Unsupervised and Robust Cross-Lingual Word Embedding using Domain Flow Interpolation.



## Requirements

To install requirements:

`pip install -r requirements.txt`


## Get Datasets

### Get Monolingual Word Embeddings

Download monolingual word embeddings from [FastText Embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html). </br>
You can download the English (en) and Spanish (es) embeddings by running:
```bash
cd data/fastText/
# English fastText Wikipedia embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
# Spanish fastText Wikipedia embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
```

You can also download all tested embeddings by running:
```bash
./get_monolingual_embeddings.sh
```

### Get evaluation datasets

Download [bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) from MUSE project.
You can simply run:
```bash
cd data/dictionaries/
# English fastText Wikipedia embeddings
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.5000-6500.txt
# Spanish fastText Wikipedia embeddings
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.5000-6500.txt
```
You can also download all dictionaries tested in the paper by running:
```bash
./get_evaluation.sh
```


## How to run

The general command: </br>
`CUDA_VISIBLE_DEVICES=<gpu-id> python  unsupervised.py  --src_lang <source-language> --tgt_lang <target-language> --src_emb <source-embedding-path> --tgt_emb <target-embedding-path> --max_vocab_A <max-vocab-size-source> --max_vocab_B <max-vocab-size-target> --dis_most_frequent_AB <most-freq-emb-src2tgt-adversary> --dis_most_frequent_BA <most-freq-emb-tgt2src-adversary> --normalize_embeddings <normalizing-values> --emb_dim_autoenc <code-space-dimension> --dis_lambda <adversarial-loss-weight> --cycle_lambda <cycle-loss-weight> --reconstruction_lambda <reconstruction-loss-weight> --finetune_epochs <number-of-finetune-epochs>`

You can also control other hyperparameters. </br></br>
For example to run EN-ES:

```bash
CUDA_VISIBLE_DEVICES=0 python unsupervised.py --src_lang en --tgt_lang es --src_emb "./data/fastText/wiki.en.vec" --tgt_emb "./data/fastText/wiki.es.vec" --mid_domain True --n_epochs 20 --autoenc_epochs 25 --epoch_size 100000 --dico_eval "./data/dictionaries/" --finetune_epochs 5 
```

You can also simply run:

```bash
./run.sh
```

You will get the word translation accuracies at different precision (1, 5, 10) for EN-ES and ES-EN.

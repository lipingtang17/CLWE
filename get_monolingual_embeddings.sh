## Download all fastText monolingual word embeddings
cd data/fastText/
for lg in en es de it ms ar he ja zh
do
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$lg.vec
done
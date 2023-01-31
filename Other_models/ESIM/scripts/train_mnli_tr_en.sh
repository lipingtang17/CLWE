cd preprocessing/

python preprocess_mnli.py \
--target_dir "../../data/preprocessed/MNLI/vecmap/en_es" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/en_es/en.vec"

#cd ../../data/preprocessed/MNLI/vecmap/en_es/
#rm *data.pkl
#rm worddict.pkl
cd ../training
python train_mnli.py \
--device "cuda:6" \
--embeddings "../../data/preprocessed/MNLI/vecmap/en_es/embeddings.pkl" \
--target_dir "../../data/checkpoints/MNLI_fixed/vecmap/en_es/"

## test en on XNLI (suc)
cd ../preprocessing/
python preprocess_xnli.py \
--target_dir "../../data/preprocessed/XNLI/vecmap/en_es" \
--lang "en" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/en_es/en.vec"

cd ../testing
python test_xnli.py \
--test_data "../../data/preprocessed/XNLI/vecmap/en_es/test_data.pkl" \
--checkpoint "../../data/checkpoints/MNLI_fixed/vecmap/en_es/best.pth.tar" \
--embeddings_file "../../data/preprocessed/XNLI/vecmap/en_es/embeddings.pkl"

## test tr on XNLI (suc)
cd ../preprocessing/
python preprocess_xnli.py \
--target_dir "../../data/preprocessed/XNLI/vecmap/es" \
--lang "es" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/en_es/es.vec"

cd ../testing
python test_xnli.py \
--test_data "../../data/preprocessed/XNLI/vecmap/es/test_data.pkl" \
--checkpoint "../../data/checkpoints/MNLI_fixed/vecmap/en_es/best.pth.tar" \
--embeddings_file "../../data/preprocessed/XNLI/vecmap/es/embeddings.pkl"



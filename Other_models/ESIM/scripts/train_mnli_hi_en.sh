#cd preprocessing/
#
#python preprocess_mnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_hi/011" \
#--target_dir "../../data/preprocessed/MNLI/hi_en"
#
#cd ../../data/preprocessed/MNLI/hi_en/
#rm *data.pkl
#rm worddict.pkl
#cd ../../../../scripts/training
#python train_mnli.py \
#--device "cuda:3" \
#--embeddings "../../data/preprocessed/MNLI/hi_en/embeddings.pkl" \
#--target_dir "../../data/checkpoints/MNLI_fixed/hi_en"


### test en on XNLI (suc)
#cd preprocessing/
#python preprocess_xnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_hi/011" \
#--target_dir "../../data/preprocessed/XNLI/hi_en" \
#--lang "en" \
#--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec"
#
#cd ../testing
#python test_xnli.py \
#--test_data "../../data/preprocessed/XNLI/hi_en/test_data.pkl" \
#--checkpoint "../../data/checkpoints/MNLI_fixed/hi_en/best.pth.tar" \
#--embeddings_file "../../data/preprocessed/XNLI/hi_en/embeddings.pkl"

## test tr on XNLI (suc)
cd preprocessing/
python preprocess_xnli.py \
--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_hi/011" \
--target_dir "../../data/preprocessed/XNLI/hi" \
--lang "hi" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.hi.vec"

cd ../testing
python test_xnli.py \
--test_data "../../data/preprocessed/XNLI/hi/test_data.pkl" \
--checkpoint "../../data/checkpoints/MNLI_fixed/hi_en/best.pth.tar" \
--embeddings_file "../../data/preprocessed/XNLI/hi/embeddings.pkl"

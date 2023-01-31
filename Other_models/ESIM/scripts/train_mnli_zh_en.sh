#cd preprocessing/
#
#python preprocess_mnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_zh/015" \
#--target_dir "../../data/preprocessed/MNLI/zh_en"
#
#cd ../../data/preprocessed/MNLI/zh_en/
#rm *data.pkl
#rm worddict.pkl
#cd ../../../../scripts/training
#python train_mnli.py \
#--device "cuda:4" \
#--embeddings "../../data/preprocessed/MNLI/zh_en/embeddings.pkl" \
#--target_dir "../../data/checkpoints/MNLI_fixed/zh_en"


### test en on XNLI (suc)
#cd preprocessing/
#python preprocess_xnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_zh/015" \
#--target_dir "../../data/preprocessed/XNLI/zh_en" \
#--lang "en" \
#--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec"
#
#cd ../testing
#python test_xnli.py \
#--test_data "../../data/preprocessed/XNLI/zh_en/test_data.pkl" \
#--checkpoint "../../data/checkpoints/MNLI_fixed/zh_en/best.pth.tar" \
#--embeddings_file "../../data/preprocessed/XNLI/zh_en/embeddings.pkl"

## test tr on XNLI (suc)
cd preprocessing/
python preprocess_xnli.py \
--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_zh/011" \
--target_dir "../../data/preprocessed/XNLI/zh" \
--lang "zh" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.zh.vec"

cd ../testing
python test_xnli.py \
--test_data "../../data/preprocessed/XNLI/zh/test_data.pkl" \
--checkpoint "../../data/checkpoints/MNLI_fixed/zh_en/best.pth.tar" \
--embeddings_file "../../data/preprocessed/XNLI/zh/embeddings.pkl"
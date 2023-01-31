#cd preprocessing/
#
#python preprocess_mnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/orig/en_ru/001" \
#--target_dir "../../data/preprocessed/MNLI/ru_en"
#
#cd ../../data/preprocessed/MNLI/ru_en/
#rm *data.pkl
#rm worddict.pkl
#cd ../../../../scripts/training
#python train_mnli.py \
#--device "cuda:3" \
#--embeddings "../../data/preprocessed/MNLI/ru_en/embeddings.pkl" \
#--target_dir "../../data/checkpoints/MNLI_fixed/ru_en"

cd preprocessing

python preprocess_xnli.py \
--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_zh/015" \
--target_dir "../../data/preprocessed/XNLI/zh" \
--config "../../config/preprocessing/xnli_preprocessing/zh_preprocessing.json"

cd ../testing
#python test_xnli.py \
#--test_data "../../data/preprocessed/XNLI/ru/test_data.pkl" \
#--checkpoint "../../data/checkpoints/MNLI_fixed/ru_en/best.pth.tar" \
#--embeddings_file "../../data/preprocessed/XNLI/ru/embeddings.pkl"

python test_xnli.py \
--test_data "../../data/preprocessed/XNLI/zh/test_data.pkl" \
--checkpoint "../../data/checkpoints/MNLI_fixed/zh_en/best.pth.tar" \
--embeddings_file "../../data/preprocessed/XNLI/zh/embeddings.pkl"
cd preprocessing/

python preprocess_mnli.py \
--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/orig/en_hi/011" \
--target_dir "../../data/preprocessed/MNLI/hi_en_fail"

cd ../../data/preprocessed/MNLI/hi_en_fail/
rm *data.pkl
rm worddict.pkl
cd ../../../../scripts/training
python train_mnli.py \
--device "cuda:6" \
--embeddings "../../data/preprocessed/MNLI/hi_en_fail/embeddings.pkl" \
--target_dir "../../data/checkpoints/MNLI_fixed/hi_en_fail"
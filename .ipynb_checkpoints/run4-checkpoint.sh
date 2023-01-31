export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

exp_name_en='MUSE/tr_003_en'
exp_name_z='MUSE/tr_003'
tgt_lang='tr'
mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/MUSE/en_tr/003"
exp_id=003

# ## generate the best mapping after procrustes
# CUDA_VISIBLE_DEVICES=1 python Other_models/MUSE/unsupervised.py \
# --n_epochs 0 \
# --n_refinement 1 \
# --src_lang en \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/MUSE/" \
# --exp_name "en_$tgt_lang" \
# --exp_id "$exp_id" \
# --export ""


## preprocess (en) for (MNLI) training
echo -e "\n ** preprocess (en) for (MNLI) training **"
cd Other_models/ESIM/scripts/preprocessing/
rm -r /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/
python preprocess_mnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en
rm /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/*data.pkl
rm /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/worddict.pkl

## train (en) on (MNLI)
echo -e "\n ** train (en) on (MNLI) data **"
cd ../training
python train_mnli.py \
--device "cuda:4" \
--embeddings /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/embeddings.pkl \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en

## preprocess (en) for (XNLI) testing
echo -e "\n ** preprocess (en) for (XNLI) testing **"
cd ../preprocessing/
rm -rf /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en
python preprocess_xnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en \
--lang "en" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec"

## test (en) on (XNLI)
echo -e "\n ** test (en) on (XNLI) **"
cd ../testing
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en/embeddings.pkl

## preprocess (tgt_lang) for (XNLI) testing
echo -e "\n ** preprocess ($tgt_lang) for (XNLI) testing **"
cd ../preprocessing
rm -rf /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z
python preprocess_xnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z \
--lang $tgt_lang \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec

## test (tgt_lang) on (XNLI)
echo -e "\n ** test ($tgt_lang) on (XNLI) **"
cd ../testing
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/embeddings.pkl



export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

exp_name_en='proc/ar_mid204_l_en'
exp_name_z='proc/ar_mid204_l'
tgt_lang='ar'
mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_ar/204"
exp_id=204

## generate the best mapping after procrustes
CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang en \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/" \
--exp_name "mid/en_$tgt_lang" \
--exp_id "$exp_id" \
--map_init "identity" \
--finetune_epochs 0 \
--export_emb False \
--save_proc True \
--n_procrustes 100 \
--n_symmetric_reweighting 100

## preprocess (en) for (MNLI) training
echo -e "\n ** preprocess (en) for (MNLI) training **"
cd Other_models/ESIM/scripts/preprocessing/
rm -r /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/
python preprocess_mnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en
# rm /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/*data.pkl
# rm /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/worddict.pkl

## train (en) on (MNLI)
echo -e "\n ** train (en) on (MNLI) data **"
cd ../training
python train_mnli.py \
--device "cuda:4" \
--embeddings /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/embeddings.pkl \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en \
--train_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/train_data.pkl \
--valid_file_matched /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/matched_dev_data.pkl \
--valid_file_mismatched /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/mismatched_dev_data.pkl

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


exp_name_en='proc/ar_mid201_l_en'
exp_name_z='proc/ar_mid201_l'
tgt_lang='ar'
mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_ar/201"
exp_id=201

cd ../../../../

## generate the best mapping after procrustes
CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang en \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/" \
--exp_name "mid/en_$tgt_lang" \
--exp_id "$exp_id" \
--map_init "identity" \
--finetune_epochs 0 \
--export_emb False \
--save_proc True \
--n_procrustes 100 \
--n_symmetric_reweighting 100

## preprocess (en) for (MNLI) training
echo -e "\n ** preprocess (en) for (MNLI) training **"
cd Other_models/ESIM/scripts/preprocessing/
rm -r /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/
python preprocess_mnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en
# rm /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/*data.pkl
# rm /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/worddict.pkl

## train (en) on (MNLI)
echo -e "\n ** train (en) on (MNLI) data **"
cd ../training
python train_mnli.py \
--device "cuda:4" \
--embeddings /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/embeddings.pkl \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en \
--train_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/train_data.pkl \
--valid_file_matched /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/matched_dev_data.pkl \
--valid_file_mismatched /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/mismatched_dev_data.pkl

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


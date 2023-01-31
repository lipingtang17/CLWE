export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

exp_name_en='vecmap/zh_en'
exp_name_z='vecmap/zh'
tgt_lang='zh'
exp_dir='en_zh'

## preprocess (en) for (MNLI) training
echo -e "\n ** preprocess (en) for (MNLI) training **"
cd Other_models/ESIM/scripts/preprocessing/
rm -r /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/MNLI/$exp_name_en/
python preprocess_mnli.py \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/$exp_dir/en.vec \
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
#rm -rf /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en
python preprocess_xnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--lang "en"

## test (en) on (XNLI)
echo -e "\n ** test (en) on (XNLI) **"
cd ../testing
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_en/embeddings.pkl

## preprocess (ru) for (XNLI) testing
echo -e "\n ** preprocess (zh) for (XNLI) testing **"
cd ../preprocessing
rm -rf /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z
python preprocess_xnli.py \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/$exp_dir/$tgt_lang.vec \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z \
--lang $tgt_lang

## test (tr) on (XNLI)
echo -e "\n ** test (zh) on (XNLI) **"
cd ../testing
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/embeddings.pkl
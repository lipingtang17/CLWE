#!/bin/bash

export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

# exp_name_en='proc/hi_mid204_en'
# exp_name_z='proc/hi_mid204'
# tgt_lang='hi'
# mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_hi/204"
# exp_id=204

exp_name_en='orig/de_006_en'
exp_name_z='orig/de_006'
tgt_lang='de'
mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_orig/en_de/006"
exp_id=006

cd Other_models/ESIM/scripts/testing

## preprocess (tgt_lang) for (XNLI) testing
echo -e "\n ** preprocess ($tgt_lang) for (XNLI) testing **"
cd ../preprocessing
rm -rf /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z
python preprocess_xnli.py \
--mapping_dir $mapping_dir \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z \
--lang $tgt_lang \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec


cd ../testing
## test (tgt_lang) on (XNLI)
for((i=20;i<=26;i++));  
do   
echo -e "\n ** test ($tgt_lang) on (XNLI) using /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/esim_$i.pth.tar**"
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/esim_$i.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/embeddings.pkl
done

# echo -e "\n ** test ($tgt_lang) on (XNLI) using /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar**"
# python test_xnli.py \
# --test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/test_data.pkl \
# --checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar \
# --embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/embeddings.pkl

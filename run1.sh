#!/bin/bash

export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

# exp_name_en='orig/hi_001_l_en'
# exp_name_z='orig/hi_001_l'
# tgt_lang='hi'
# mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_orig/en_hi/001"
# exp_id=001

declare -a tgt_lang_list=(ar bg hi tr fr ru es de)
declare -a exp_id_llist=(002 002 002 002 002 002 002 002)
declare -a exp_name_en_list
declare -a exp_name_z_list
declare -a mapping_dir_list


for((i=0;i<=7;i++));  
do   
tgt_lang=${tgt_lang_list[$i]}
exp_id=${exp_id_llist[$i]}
exp_name_en_list[$i]='MUSE/'$tgt_lang'_'$exp_id'_l_en'
exp_name_z_list[$i]='MUSE/'$tgt_lang'_'$exp_id'_l'
mapping_dir_list[$i]='/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/MUSE/en_'$tgt_lang'/'$exp_id
done



cd Other_models/ESIM/scripts/testing


output_file="/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_muse.txt"

for((c=0;c<=7;c++));
do

tgt_lang=${tgt_lang_list[$c]}
exp_id=${exp_id_llist[$c]}
exp_name_en=${exp_name_en_list[$c]}
exp_name_z=${exp_name_z_list[$c]}
mapping_dir=${mapping_dir_list[$c]}

# test (tgt_lang) on (XNLI)
for((i=1;i<=3;i++)); 
do

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
echo -e "\n ** test ($tgt_lang) on (XNLI) using /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar**"
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/best.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/embeddings.pkl \
--iteration $i \
--output_file $output_file

for((j=1;j<=20;j++));  
do   
echo -e "\n ** test ($tgt_lang) on (XNLI) using /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/esim_$j.pth.tar**"
python test_xnli.py \
--test_data /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/test_data.pkl \
--checkpoint /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/checkpoints/MNLI/$exp_name_en/esim_$j.pth.tar \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z/embeddings.pkl \
--iteration $i \
--output_file $output_file
done
done
done


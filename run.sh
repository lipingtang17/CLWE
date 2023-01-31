#!/bin/bash

export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

# exp_name_en='orig/hi_001_l_en'
# exp_name_z='orig/hi_001_l'
# tgt_lang='hi'
# mapping_dir="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_orig/en_hi/001"
# exp_id=001

declare -a tgt_lang_list=(ar ar bg bg fr ru)
declare -a exp_id_llist=(204 201 002 102 001 011)
# declare -a tgt_lang_list=(hi tr fr ru de es zh)
# declare -a exp_id_llist=(011 011 001 011 001 001 019)
declare -a exp_name_en_list
declare -a exp_name_z_list
declare -a mapping_dir_list


for((i=0;i<=5;i++));  
do   
tgt_lang=${tgt_lang_list[$i]}
exp_id=${exp_id_llist[$i]}
exp_name_en_list[$i]='proc/'$tgt_lang'_mid'$exp_id'_l_en'
exp_name_z_list[$i]='proc/'$tgt_lang'_mid'$exp_id'_l'
mapping_dir_list[$i]='/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/mid/en_'$tgt_lang'/'$exp_id
done

# for((i=3;i<=8;i++));  
# do   
# tgt_lang=${tgt_lang_list[$i]}
# exp_id=${exp_id_llist[$i]}
# exp_name_en_list[$i]="orig_f/"$tgt_lang"_"$exp_id"_en"
# exp_name_z_list[$i]="orig_f/"$tgt_lang"_"$exp_id
# mapping_dir_list[$i]="/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/orig/en_"$tgt_lang"/"$exp_id
# done


# ## generate the best mapping after procrustes
# CUDA_VISIBLE_DEVICES=3 python unsupervised.py \
# --src_lang en \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --mid_domain False \
# --n_epochs 0 \
# --autoenc_epochs 0 \
# --epoch_size 100000 \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/" \
# --exp_name "orig/en_$tgt_lang" \
# --exp_id "$exp_id" \
# --map_init "identity" \
# --finetune_epochs 0 \
# --export_emb False \
# --save_proc True \
# --n_procrustes 100 \
# --n_symmetric_reweighting 100

cd Other_models/ESIM/scripts/testing


output_file="/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_proc.txt"

for((c=0;c<=5;c++));
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


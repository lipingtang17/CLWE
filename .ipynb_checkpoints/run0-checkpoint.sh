#!/bin/bash

export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"


## vecmap

declare -a tgt_lang_list=(de ru)
declare -a exp_name_en_list
declare -a exp_name_z_list
declare -a exp_dir_list


for((i=0;i<=1;i++));  
do   
tgt_lang=${tgt_lang_list[$i]}
exp_name_en_list[$i]="vecmap/"$tgt_lang"_en_l"
exp_name_z_list[$i]="vecmap/"$tgt_lang"_l"
exp_dir_list[$i]="en_"$tgt_lang
done



cd Other_models/ESIM/scripts/testing


output_file="/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_vecmap.txt"

for((c=0;c<=1;c++));
do

tgt_lang=${tgt_lang_list[$c]}
exp_name_en=${exp_name_en_list[$c]}
exp_name_z=${exp_name_z_list[$c]}
exp_dir=${exp_dir_list[$c]}

# test (tgt_lang) on (XNLI)
for((i=1;i<=3;i++)); 
do

## preprocess (tgt_lang) for (XNLI) testing
echo -e "\n ** preprocess ($tgt_lang) for (XNLI) testing **"
cd ../preprocessing
rm -rf /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z
python preprocess_xnli.py \
--embeddings_file /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/vecmap/$exp_dir/$tgt_lang.vec \
--target_dir /mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/ESIM/data/preprocessed/XNLI/$exp_name_z \
--lang $tgt_lang

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


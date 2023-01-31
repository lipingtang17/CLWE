export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

# tgt_lang='es'
# CUDA_VISIBLE_DEVICES=2 python unsupervised.py \
# --src_lang en \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --mid_domain False \
# --n_epochs 10 \
# --autoenc_epochs 25 \
# --epoch_size 100000 \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_name "orig/en_$tgt_lang" \
# --exp_id "001" \
# --map_init "identity" \
# --finetune_epochs 0 \
# --save_proc False


src_lang='fi'
tgt_lang='et'
CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "identical/"$src_lang"_"$tgt_lang \
--exp_id "001" \
--map_init "id_char" \
--finetune_epochs 0 \
--save_proc False
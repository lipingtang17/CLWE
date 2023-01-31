export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

# src_lang='en'
# tgt_lang='fi'
# CUDA_VISIBLE_DEVICES=0 python unsupervised.py \
# --src_lang $src_lang \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --mid_domain True \
# --n_epochs 0 \
# --autoenc_epochs 0 \
# --epoch_size 100000 \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_name "mid/"$src_lang"_"$tgt_lang \
# --exp_id "001" \
# --map_init "identity" \
# --finetune_epochs 3 \
# --save_proc False \

# src_lang='en'
# tgt_lang='it'
# CUDA_VISIBLE_DEVICES=1 python unsupervised.py \
# --src_lang $src_lang \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --mid_domain True \
# --n_epochs 0 \
# --autoenc_epochs 0 \
# --epoch_size 100000 \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_name "vecmap_it" \
# --exp_id "it_mid_001" \
# --map_init "identity" \
# --finetune_epochs 3 \
# --save_proc False \
# --exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"



# src_lang='en'
# tgt_lang='it'
# CUDA_VISIBLE_DEVICES=1 python unsupervised.py \
# --src_lang $src_lang \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --mid_domain True \
# --n_epochs 0 \
# --autoenc_epochs 0 \
# --epoch_size 100000 \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_name "vecmap_it" \
# --exp_id "it_orig_001" \
# --map_init "identity" \
# --finetune_epochs 3 \
# --save_proc False \
# --exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

# src_lang='en'
# tgt_lang='it'
# CUDA_VISIBLE_DEVICES=1 python unsupervised.py \
# --src_lang $src_lang \
# --tgt_lang $tgt_lang \
# --src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
# --tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
# --mid_domain True \
# --n_epochs 0 \
# --autoenc_epochs 0 \
# --epoch_size 100000 \
# --dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
# --exp_name "vecmap_it" \
# --exp_id "it_orig_002" \
# --map_init "identity" \
# --finetune_epochs 3 \
# --save_proc False \
# --exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

src_lang='en'
tgt_lang='it'
CUDA_VISIBLE_DEVICES=1 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_it" \
--exp_id "it_orig_003" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

src_lang='en'
tgt_lang='it'
CUDA_VISIBLE_DEVICES=1 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_it" \
--exp_id "it_orig_004" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

src_lang='en'
tgt_lang='it'
CUDA_VISIBLE_DEVICES=1 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_it" \
--exp_id "it_orig_005" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"


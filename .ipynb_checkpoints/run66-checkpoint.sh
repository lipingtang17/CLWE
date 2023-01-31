export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"

src_lang='en'
tgt_lang='fi'
CUDA_VISIBLE_DEVICES=7 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain False \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_fi" \
--exp_id "fi_orig_301" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

src_lang='en'
tgt_lang='fi'
CUDA_VISIBLE_DEVICES=7 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain False \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_fi" \
--exp_id "fi_orig_303" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

src_lang='en'
tgt_lang='fi'
CUDA_VISIBLE_DEVICES=7 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain False \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_fi" \
--exp_id "fi_orig_309" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

src_lang='en'
tgt_lang='fi'
CUDA_VISIBLE_DEVICES=7 python unsupervised.py \
--src_lang $src_lang \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$src_lang.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain False \
--n_epochs 0 \
--autoenc_epochs 0 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "vecmap_fi" \
--exp_id "fi_orig_310" \
--map_init "identity" \
--finetune_epochs 3 \
--save_proc False \
--exp_path "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/unsup/dumped/"

export PYTHONPATH="./Other_models/ESIM/":"${PYTHONPATH}"
export PYTHONPATH="../../":"${PYTHONPATH}"



tgt_lang='zh'

CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang en \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 50 \
--autoenc_epochs 25 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "mid/en_$tgt_lang" \
--exp_id "104" \
--map_init "identity" \
--finetune_epochs 10 \
--save_proc True

CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang en \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 50 \
--autoenc_epochs 25 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "mid/en_$tgt_lang" \
--exp_id "105" \
--map_init "identity" \
--finetune_epochs 10 \
--save_proc True

CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang en \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 50 \
--autoenc_epochs 25 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "mid/en_$tgt_lang" \
--exp_id "106" \
--map_init "identity" \
--finetune_epochs 10 \
--save_proc True


tgt_lang='hi'

CUDA_VISIBLE_DEVICES=4 python unsupervised.py \
--src_lang en \
--tgt_lang $tgt_lang \
--src_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec" \
--tgt_emb "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.$tgt_lang.vec" \
--mid_domain True \
--n_epochs 50 \
--autoenc_epochs 25 \
--epoch_size 100000 \
--dico_eval "/mnt/ssd-201-112-01/cpii.local/lptang/dictionaries/MUSE/" \
--exp_name "mid/en_$tgt_lang" \
--exp_id "104" \
--map_init "identity" \
--finetune_epochs 10 \
--save_proc True

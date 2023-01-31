#cd preprocessing/
#
#python preprocess_mnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/orig/en_tr/002" \
#--target_dir "../../data/preprocessed/MNLI/tr_en_fail"
#
#cd ../../data/preprocessed/MNLI/tr_en_fail/
#rm *data.pkl
#rm worddict.pkl
#cd ../../../../scripts/training
#python train_mnli.py \
#--device "cuda:7" \
#--embeddings "../../data/preprocessed/MNLI/tr_en_fail/embeddings.pkl" \
#--target_dir "../../data/checkpoints/MNLI_fixed/tr_en_fail"


### test en on XNLI (fail)
#cd preprocessing/
#python preprocess_xnli.py \
#--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/orig/en_tr/002" \
#--target_dir "../../data/preprocessed/XNLI/tr_en_fail" \
#--lang "en" \
#--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.en.vec"
#
#cd ../testing
#python test_xnli.py \
#--test_data "../../data/preprocessed/XNLI/tr_en_fail/test_data.pkl" \
#--checkpoint "../../data/checkpoints/MNLI_fixed/tr_en_fail/best.pth.tar" \
#--embeddings_file "../../data/preprocessed/XNLI/tr_en_fail/embeddings.pkl"

## test tr on XNLI (fail)
cd preprocessing/
python preprocess_xnli.py \
--mapping_dir "/mnt/ssd-201-112-01/cpii.local/lptang/checkpoints/CLWE/unsup_formal/orig/en_tr/002" \
--target_dir "../../data/preprocessed/XNLI/tr_fail" \
--lang "tr" \
--embeddings_file "/mnt/ssd-201-112-01/cpii.local/lptang/WordEmb/fasttext/wiki.tr.vec"

cd ../testing
python test_xnli.py \
--test_data "../../data/preprocessed/XNLI/tr_fail/test_data.pkl" \
--checkpoint "../../data/checkpoints/MNLI_fixed/tr_en_fail/best.pth.tar" \
--embeddings_file "../../data/preprocessed/XNLI/tr_fail/embeddings.pkl"
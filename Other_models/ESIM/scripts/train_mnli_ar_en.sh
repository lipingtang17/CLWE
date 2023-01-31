cd preprocessing/
python preprocess_mnli.py --config "/users/cpii.local/lptang/NLP_models/CLWE_formal/Other_models/ESIM/config/preprocessing/mnli_preprocessing_en/zh.json"
cd ../../data/preprocessed/MNLI/zh_en/
rm *data.pkl
rm worddict.pkl
cd ../../../../scripts/training
python train_mnli.py --device "cuda:4" --embeddings "../../data/preprocessed/MNLI/zh_en/embeddings.pkl" --target_dir "../../data/checkpoints/MNLI_fixed/zh_en"
import numpy as np
from collections import defaultdict

result_inputs = ["/users/cpii.local/lptang/NLP_models/CLWE_formal/output/results.txt"]
result_output_best = "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/results_avg_best.txt"
result_output_high = "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/results_avg_high.txt"

lines = []
for result_input in result_inputs:
    with open(result_input, "r", encoding="utf-8") as f1:
        lines += f1.readlines()
        
acc_dict_best = defaultdict(list)
acc_dict_high = defaultdict(list)
for line in lines:
    tgt_lang, model, emb_ckpts, iteration, esim_epoch, accuracy = line.split("\t")
    if esim_epoch == "best":
        acc_dict_best[tgt_lang+"_"+model].append(float(accuracy))
    else:
        acc_dict_high[tgt_lang+"_"+model].append(float(accuracy))

with open(result_output_best, "a", encoding="utf-8") as f2:
    for key, value in sorted(acc_dict_best.items(), key=lambda item: item[0]):
        tgt_lang, model = key.split("_", 1)
        acc_avg = np.mean(value)
        f2.write(tgt_lang+"\t"+model+"\t"+str(acc_avg)+"\n")

with open(result_output_high, "a", encoding="utf-8") as f2:
    for key, value in sorted(acc_dict_high.items(), key=lambda item: item[0]):
        tgt_lang, model = key.split("_", 1)
        acc_avg = np.mean(value)
        f2.write(tgt_lang+"\t"+model+"\t"+str(acc_avg)+"\n")

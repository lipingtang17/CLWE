result_inputs = ["/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_proc.txt",
                 "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_origf.txt",
                 "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_orig.txt",
                 "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_vecmap.txt", 
                 "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/original_results_muse.txt"
                ]
result_output = "/users/cpii.local/lptang/NLP_models/CLWE_formal/output/results.txt"

lines = []
for result_input in result_inputs:
    with open(result_input, "r", encoding="utf-8") as f1:
        lines += f1.readlines()
with open(result_output, "a", encoding="utf-8") as f2:
    iteration_prev = 0
    acc_dict = {}
    acc_max = 0
    for line in lines:
        tgt_lang, model, emb_ckpts, iteration, esim_epoch, accuracy = line.split("\t")
        if esim_epoch == "best":
            f2.write(tgt_lang+"\t"+model+"\t"+emb_ckpts+"\t"+iteration+"\t"+esim_epoch+"\t"+accuracy)
    for i, line in enumerate(lines):
        tgt_lang, model, emb_ckpts, iteration, esim_epoch, accuracy = line.split("\t")
        if esim_epoch == "best" or i == len(lines)-1:
            if acc_dict:
                acc_max = max(acc_dict.values())
                esim_epoch_max = max(acc_dict, key=acc_dict.get)
                f2.write(tgt_lang_prev+"\t"+model_prev+"\t"+emb_ckpts_prev+"\t"+iteration_prev+"\t"+esim_epoch_max+"\t"+str(acc_max)+"\n")
                acc_dict = {}
            # if i != len(lines)-1:
            #     f2.write(tgt_lang+"\t"+model+"\t"+emb_ckpts+"\t"+iteration+"\t"+esim_epoch+"\t"+accuracy)
        else:
            acc_dict[esim_epoch] = float(accuracy)
            tgt_lang_prev, model_prev, emb_ckpts_prev, iteration_prev = tgt_lang, model, emb_ckpts, iteration

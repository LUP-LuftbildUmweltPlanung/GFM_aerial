import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

def save_train_hist(train_loss, val_spa_loss, val_temp_loss, val_temp_spa_loss, val_avg, title, output_path):
    plt.figure(figsize=(18,10), layout='constrained')
    plt.plot(train_loss, color='black', linestyle='-', linewidth=3., label='Training')
    plt.plot(val_spa_loss, color='tab:blue', linestyle='-',linewidth=3., label='Spatially independent validation')
    plt.plot(val_temp_loss, color='tab:orange', linestyle='-',linewidth=3., label='Temporally independent validation')
    plt.plot(val_temp_spa_loss, color='tab:green', linestyle='-',linewidth=3., label='Spatiotemporally independent validation')
    if val_avg.empty:
        val_avg = (val_spa_loss + val_temp_loss + val_temp_spa_loss) / 3
    plt.plot(val_avg, color='tab:red', linestyle='--',linewidth=3., label='Validation average')

    plt.title(title, fontsize=45)
    plt.xlabel("Epochs",fontsize=40)
    plt.ylabel("Combined Loss", fontsize=40)
    plt.legend(fontsize=37)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    #plt.show()
    plt.savefig(output_path, dpi=400, format="pdf")

def save_train_hist_mult_models(train_losses, val_avg_losses, title, output_path, colors, models):
    plt.figure(figsize=(18,10), layout='constrained')

    print(len(train_losses))
    print(len(val_avg_losses))
    print(len(models))
    print(len(colors))

    for i in range(len(train_losses)):
        plt.plot(train_losses[i], color=colors[i], linestyle='-', linewidth=3., label=f'Training {models[i]}')
    for i in range(len(val_avg_losses)):
        plt.plot(val_avg_losses[i], color=colors[i], linestyle='--',linewidth=3., label=f'Validation average {models[i]}')


    #plt.title(title, fontsize=45)
    plt.xlabel("Epochs",fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()
    #plt.savefig(output_path, dpi=400, format="pdf")

def process_single_csv(csv_path, title, output_path):
    if csv_path.endswith(".csv") and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    else:
        return
    save_train_hist(df["train_loss"], df["val_loss_spa"], df["val_loss_temp"], df["val_loss_temp_spa"], df["val_loss_avg"], title, output_path)

def process_multiple_csv(csv_paths, title, item, output_path):
    elem_dict = {}
    for elem in csv_paths:
        if elem.endswith(".csv") and Path.is_file(Path(elem)):
            name = Path(elem).stem
            df = pd.read_csv(elem, on_bad_lines='skip')
            elem_dict[name] = df[item]
        else:
            continue

    save_train_hist(elem_dict["train_log_table"], elem_dict["vali_log_table_spa_ind"], elem_dict["vali_log_table_temp_ind"], elem_dict["vali_log_table_temp_spa_ind"], pd.DataFrame(), title, output_path)

def process_multiple_csv_multi_models(csv_paths, models, colors, out_path, title):
    train_list = []
    vali_list = []
    for i in range(len(csv_paths)):
        if csv_paths[i].endswith(".csv") and Path.is_file(Path(csv_paths[i])):
            df = pd.read_csv(csv_paths[i], on_bad_lines='skip')
            train_list.append(df["train_loss"])
            vali_list.append(df["val_loss_avg"])
        else:
            continue

    save_train_hist_mult_models(train_list, vali_list, title, out_path, colors, models)

# Comparison of avg validation losses and training losses of all GFM-aerial models:
all_models_title = "Combined losses during training of GFM-aerial models"
colors = ["blue","orange","green"]
models = ["GFM-aerial_192_swin", "GFM-aerial_192_gfm", "GFM-aerial_384_gfm"]
out_file = (r"gfm_aerial_train_hist_comparison_new.pdf")                                    #modify, if necessary
csv_files = [r"gfmaerial_pretrain_swin_teacher\gfm_aerial_swin_teacher_192imgSize\stats\loss_log_table.csv",            #modify, if necessary
             r"gfmaerial_pretrain_gfm_teacher\gfm_aerial_gfm_teacher_192imgSize\stats_gfm_grad_acc\loss_log_table.csv", #modify, if necessary
             r"gfmaerial_pretrain_gfm_teacher\gfm_aerial_gfm_teacher\stats_gfm_grad_acc\loss_log_table.csv"]            #modify, if necessary

process_multiple_csv_multi_models(csv_files, models, colors, out_file, all_models_title)



# All combined losses of one model from loss_log_table.csv:
out_file = (r"gfm_aerial_192_gfm_train_hist.pdf")    #modify, if necessary
all_losses_title = "GFM-aerial-192-gfm: Loss during training"
process_single_csv(r"gfm_aerial_gfm_teacher_192imgSize\stats_gfm_grad_acc\loss_log_table.csv", all_losses_title, out_file)    #modify, if necessary


# Reconstruction losses of one model:
out_file = (r"gfm_aerial_192_gfm_train_hist.pdf")    #modify, if necessary
csv_paths = [r"gfmaerial_pretrain_gfm_teacher\gfm_aerial_gfm_teacher_192imgSize\stats_gfm_grad_acc\train_log_table.csv",            #modify, if necessary
            r"gfmaerial_pretrain_gfm_teacher\gfm_aerial_gfm_teacher_192imgSize\stats_gfm_grad_acc\vali_log_table_spa_ind.csv",      #modify, if necessary
            r"gfmaerial_pretrain_gfm_teacher\gfm_aerial_gfm_teacher_192imgSize\stats_gfm_grad_acc\vali_log_table_temp_ind.csv",     #modify, if necessary
            r"gfmaerial_pretrain_gfm_teacher\gfm_aerial_gfm_teacher_192imgSize\stats_gfm_grad_acc\vali_log_table_temp_spa_ind.csv"  #modify, if necessary
             ]

all_losses_title = "Reconstruction losses during training"
process_multiple_csv(csv_paths, all_losses_title, "reconstruction_loss_avg", out_file)
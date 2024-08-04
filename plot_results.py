import h5py

import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot


os.chdir(os.path.join(os.path.dirname(__file__), 'PFLlib/results'))






#Open the H5 file in read mode
dr_0 = create_result(
    list_file=[
        "Cifar10_iid_FedAvg_train_c-0_3__drop-0.h5",
        "Cifar10_niid_FedAvg_train_c-0_3__drop-0.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-0__mu-0_001.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-0__mu-0_001.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-0__mu-0.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-0__mu-0.h5"
    ]
)

dr_50 = create_result(
    list_file=[
        "Cifar10_iid_FedAvg_train_c-0_3__drop-50.h5",
        "Cifar10_niid_FedAvg_train_c-0_3__drop-50.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-50__mu-0_001.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-50__mu-0_001.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-50__mu-0.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-50__mu-0.h5",
    ]
)

dr_90 = create_result(
    list_file=[
        "Cifar10_iid_FedAvg_train_c-0_3__drop-90.h5",
        "Cifar10_niid_FedAvg_train_c-0_3__drop-90.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-90__mu-0_001.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-90__mu-0_001.h5",
        "Cifar10_iid_FedProx_train_C-0_3__drop-90__mu-0.h5",
        "Cifar10_niid_FedProx_train_C-0_3__drop-90__mu-0.h5",
        #
    ]
)



def show_point(result):
  # Find the first intersection point
  intersection_index = np.argmax(np.array(result) >= 0.6)

  if intersection_index != 0:

    intersection_x = intersection_index + 1  # +1 because your x-axis starts from 1
    intersection_y = result[intersection_index]

    # Highlight the intersection point
    plt.plot(intersection_x, intersection_y, 'ro', markersize=10)

    # Add annotation
    plt.annotate(f'{intersection_x}',
                xy=(intersection_x, intersection_y),
                xytext=(intersection_x+0.3, intersection_y+0.03),
                arrowprops=dict(facecolor='black', shrink=0.05))
    

def plot_drop_rate(mode='test_acc', output_name='test_acc_plot'):
    def create_result (list_file):
        result = {}
        list_alg = ['fedavg', 'fedprox_mu_lt_0', 'fedprox_mu_0']
        for alg in list_alg:
            result[alg] = {
                'iid': {
                    'test_acc': None,
                    'train_loss': None
                },

                'niid': {
                    "test_acc": None,
                    "train_loss": None
                }
            }

        for file in list_file:
            fi_lr = file.lower()
            with h5py.File(file, 'r') as file:
            test_acc, train_loss = file['rs_test_acc'][:], file['rs_train_loss'][:]

            if 'fedavg' in fi_lr:
                if 'iid' in fi_lr and 'niid' not in fi_lr:
                result['fedavg']['iid']['test_acc'] = test_acc
                result['fedavg']['iid']['train_loss'] = train_loss
                elif 'niid' in fi_lr:
                result['fedavg']['niid']['test_acc'] = test_acc
                result['fedavg']['niid']['train_loss'] = train_loss

            elif 'fedprox' in fi_lr and '001' in fi_lr:
                if 'iid' in fi_lr and 'niid' not in fi_lr:
                result['fedprox_mu_lt_0']['iid']['test_acc'] = test_acc
                result['fedprox_mu_lt_0']['iid']['train_loss'] = train_loss
                elif 'niid' in fi_lr:
                result['fedprox_mu_lt_0']['niid']['test_acc'] = test_acc
                result['fedprox_mu_lt_0']['niid']['train_loss'] = train_loss


            elif 'fedprox' in fi_lr and 'mu-0.' in fi_lr:
                if 'iid' in fi_lr and 'niid' not in fi_lr:
                result['fedprox_mu_0']['iid']['test_acc'] = test_acc
                result['fedprox_mu_0']['iid']['train_loss'] = train_loss
                elif 'niid' in fi_lr:
                result['fedprox_mu_0']['niid']['test_acc'] = test_acc
                result['fedprox_mu_0']['niid']['train_loss'] = train_loss

        return result
    
    
    matplotlib.rc('xtick', labelsize=17)
    matplotlib.rc('ytick', labelsize=17)
    
    f = plt.figure(figsize=[40, 30])


    # Define variable

    titles = ["I.I.D Dataset", "Non-I.I.D Dataset"]
    idx_ds = ["iid", "niid"]
    rounds = [200, 200]
    mus=[1, 1]
    drop_rates=[0, 50, 90]
    sampling_rate = [1, 1]
    labels = ['FedAvg', r'FedProx ($\mu$=0)', r'FedProx ($\mu$>0)']
    list_alg = ['fedavg', 'fedprox_mu_0', 'fedprox_mu_lt_0']

    improv = 0

    for drop_rate in range(3):
        for idx in range(2):

            ax = plt.subplot(3, 2, 2*(drop_rate)+idx+1)
            ax.figure.set_figwidth(20)
            ax.figure.set_figheight(15)

            # Get test accuracy and training loss for each strategy
            list_rounds = np.arange(0, 201)

            result1 = eval("dr_" + str(drop_rates[drop_rate]) + f"['{list_alg[0]}']['{idx_ds[idx]}']['{mode}']")


            result2 = eval("dr_" + str(drop_rates[drop_rate]) + f"['{list_alg[1]}']['{idx_ds[idx]}']['{mode}']")


            result3 = eval("dr_" + str(drop_rates[drop_rate]) + f"['{list_alg[2]}']['{idx_ds[idx]}']['{mode}']")



            if mode == "test_acc":
            if drop_rate == 2 and idx == 1:
                plt.plot(list_rounds, result1, ":", linewidth=3.0, label=labels[0], color="#ff7f0e")

                show_point(result1)

                plt.plot(list_rounds, result2, '--', linewidth=3.0, label=labels[1], color="#e377c2")

                show_point(result2)

                plt.plot(list_rounds, result3, linewidth=3.0, label=labels[2], color="#17becf")

                show_point(result3)

            else:
                plt.plot(list_rounds, result1, ":", linewidth=3.0, color="#ff7f0e")

                show_point(result1)

                plt.plot(list_rounds, result2, '--', linewidth=3.0, color="#e377c2")

                show_point(result2)

                plt.plot(list_rounds, result3, linewidth=3.0, color="#17becf")

                show_point(result3)

            plt.axhline(y=0.6, color='gray', linestyle='-.', linewidth=3.0)

            elif mode == "train_loss":
            if drop_rate == 2 and idx == 1:
                plt.plot(list_rounds, result1, ":", linewidth=3.0, label=labels[0], color="#ff7f0e")
                plt.plot(list_rounds, result2, '--', linewidth=3.0, label=labels[1], color="#e377c2")
                plt.plot(list_rounds, result3, linewidth=3.0, label=labels[2], color="#17becf")

            else:
                plt.plot(list_rounds, result1, ":", linewidth=3.0, color="#ff7f0e")
                plt.plot(list_rounds, result2, '--', linewidth=3.0, color="#e377c2")
                plt.plot(list_rounds, result3, linewidth=3.0, color="#17becf")


            plt.xlabel("# Rounds", fontsize=22)
            plt.xticks(fontsize=17)
            plt.yticks(fontsize=17)

            if mode == 'train_loss' and idx == 0:
                plt.ylabel('Training Loss', fontsize=22)
            elif mode == 'test_acc' and idx == 0:
                plt.ylabel('Testing Accuracy', fontsize=22)

            if drop_rate == 0:
                plt.title(titles[idx], fontsize=22, fontweight='bold')

            ax.tick_params(color='#dddddd')
            ax.spines['bottom'].set_color('#dddddd')
            ax.spines['top'].set_color('#dddddd')
            ax.spines['right'].set_color('#dddddd')
            ax.spines['left'].set_color('#dddddd')
            ax.set_xlim(0, rounds[idx])

            ax.set_xlim(-0.1, 201)
            
    f.legend(frameon=False, loc='lower center', ncol=3, prop=dict(weight='bold'), borderaxespad=-0.3, fontsize=26)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    f.savefig(f"{output_name}.png")
    

# Define arguments
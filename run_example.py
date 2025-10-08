import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import pandas as pd
import scipy.io

from models.SkyGP_rBCM import SkyGP_rBCM
from models.SkyGP_MOE import SkyGP_MOE
from models.SkyGP_gPOE import SkyGP_gPOE
from train.SkyGP import train_SkyGP

from utils.data_loader import load_data

def plot_smse_benchmark(smse_dict, color_map=None):
    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    for label, smse in smse_dict.items():
        indices = list(range(len(smse)))
        color = color_map[label] if color_map and label in color_map else None
        plt.plot(indices, smse, linestyle='-', label=label, color=color)

    def scientific_notation(x, pos):
        if x == 0:
            return "0"
        exponent = int(np.floor(np.log10(x)))
        coeff = x / (10 ** exponent)
        return r"${:.0f} \times 10^{{{}}}$".format(coeff, exponent)

    ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()) 
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)

    plt.xlabel("Iteration")
    plt.ylabel("SMSE")
    plt.yscale("log")
    plt.title("SMSE over Iterations")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- MSLL Benchmarking ---
def plot_msll_benchmark(msll_dict, color_map=None):
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    
    for label, msll in msll_dict.items():
        indices = list(range(len(msll)))
        color = color_map[label] if color_map and label in color_map else None
        plt.plot(indices, msll, linestyle='-', label=label, color=color)
        
    ax.set_xticks([10000, 20000, 30000, 40000, 44000])
    ax.set_xticklabels(['10k', '20k', '30k', '40k', '44k'])
    plt.xlabel("Iteration")
    plt.ylabel("MSLL")
    ax.grid(True, linestyle='--', linewidth=0.7)
    plt.title("MSLL over Iterations")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
MODEL_COLOR_MAP = None

def run_all_models(
    model_types=["SkyGP_Fast", "SkyGP_Dense"],
    nearest_k_list=[1],
    dataset="sarcos",
    max_samples=50000,
    max_data_per_expert=50,
    max_experts=40,
):
    # Load hyperparameters
    X_train, Y_train, X_test, Y_test = load_data(dataset)
    if dataset == "sarcos":
        mat = scipy.io.loadmat('dataset/sarcos/Sarcos_hyper.mat')
        params = mat['params']
        lengthscale = params['lengthscale'][0, 0].flatten()
        outputscale = params['outputscale'][0, 0]
        outputscale = np.array([outputscale], dtype=np.float64) 
        noise = params['noise'][0, 0]
        noise = np.array([noise], dtype=np.float64)
    elif dataset == "pumadyn32nm":
        mat = scipy.io.loadmat('dataset/puma/pumadyn32nm_hyperparams.mat')
        dim_idx = 0
        outputscale = np.array(mat['sf'][dim_idx, 0]).astype(np.float64).squeeze()  # shape: ()
        noise       = np.array(mat['sn'][dim_idx, 0]).astype(np.float64).squeeze()  # shape: ()
        lengthscale = mat['ls'].squeeze()  #
        noise = np.array([noise], dtype=np.float64)
    elif dataset == "kin40k":
        mat_data = scipy.io.loadmat('dataset/kin40k/kin40k_hyperparams.mat')
        dim_idx = 0
        outputscale = np.array(mat_data['sf'][dim_idx, 0]).astype(np.float64).squeeze()  # shape: ()
        noise       = np.array(mat_data['sn'][dim_idx, 0]).astype(np.float64).squeeze()  # shape: ()
        lengthscale = mat_data['ls'].squeeze() 
        noise = np.array([noise], dtype=np.float64)
    elif dataset == "electric":
        mat = scipy.io.loadmat('dataset/electric/electric_hyperparams.mat')
        dim_idx = 0
        outputscale = np.array(mat['sf'][dim_idx, 0]).astype(np.float64).squeeze()  # shape: ()
        noise       = np.array(mat['sn'][dim_idx, 0]).astype(np.float64).squeeze()  # shape: ()
        lengthscale = mat['ls'].squeeze()  #
        noise = np.array([noise], dtype=np.float64)
    
    results = {}
    msll_results = {}
    timing_logs = {}

    for model_type in model_types:
        if model_type == "SkyGP_Fast":
            for k in nearest_k_list:
                print(f"\n🚀 Training SkyGP-Fast-MOE (k={k})...")
                model = SkyGP_MOE(
                    x_dim=X_train.shape[1], y_dim=1,
                    max_data_per_expert=max_data_per_expert,
                    max_experts=max_experts,
                    nearest_k=k,
                    timescale=0.04, 
                )
                smse, msll, log = train_SkyGP(
                    max_samples=max_samples,
                    dataset=dataset,
                    model=model,
                    pretrained_params=(outputscale, noise, lengthscale)
                )
                label = f"SkyGP-Fast-MOE(k={k})"
                results[label] = smse
                msll_results[label] = msll
                timing_logs[label] = log

                print(f"\n🚀 Training SkyGP-Fast-gPOE (k={k})...")
                model = SkyGP_gPOE(
                    x_dim=X_train.shape[1], y_dim=1,
                    max_data_per_expert=max_data_per_expert,
                    max_experts=max_experts,
                    nearest_k=k,
                    timescale=0.04  #timescale parameter for faster training
                )
                smse, msll, log = train_SkyGP(
                    max_samples=max_samples,
                    dataset=dataset,
                    model=model,
                    pretrained_params=(outputscale, noise, lengthscale)
                )
                label = f"SkyGP-Fast-gPOE(k={k})"
                results[label] = smse
                msll_results[label] = msll
                timing_logs[label] = log

                print(f"\n🚀 Training SkyGP-rBCM (k={k})...")
                model = SkyGP_rBCM(
                    x_dim=X_train.shape[1], y_dim=1,
                    max_data_per_expert=max_data_per_expert,
                    max_experts=max_experts,
                    nearest_k=k,
                    pretrained_params=(outputscale, noise, lengthscale),
                    timescale=0.04  #timescale parameter for faster training
                )
                smse, msll, log = train_SkyGP(
                    max_samples=max_samples,
                    dataset=dataset,
                    model=model,
                    pretrained_params=(outputscale, noise, lengthscale)
                )
                label = f"SkyGP-Fast-rBCM(k={k})"
                results[label] = smse
                msll_results[label] = msll
                timing_logs[label] = log
        elif model_type == "SkyGP_Dense":
            for k in nearest_k_list:
                print(f"\n🚀 Training SkyGP-Dense-MOE (k={k})...")
                model = SkyGP_MOE(
                    x_dim=X_train.shape[1], y_dim=1,
                    max_data_per_expert=max_data_per_expert,
                    max_experts=max_experts,
                    nearest_k=k,
                    replacement=True
                )
                smse, msll, log = train_SkyGP(
                    max_samples=max_samples,
                    dataset=dataset,
                    model=model,
                    pretrained_params=(outputscale, noise, lengthscale)
                )
                label = f"SkyGP-Dense-MOE(k={k})"
                results[label] = smse
                msll_results[label] = msll
                timing_logs[label] = log

                print(f"\n🚀 Training SkyGP-Dense-POE (k={k})...")
                model = SkyGP_gPOE(
                    x_dim=X_train.shape[1], y_dim=1,
                    max_data_per_expert=max_data_per_expert,
                    max_experts=max_experts,
                    nearest_k=k,
                    replacement=True
                )
                smse, msll, log = train_SkyGP(
                    max_samples=max_samples,
                    dataset=dataset,
                    model=model,
                    pretrained_params=(outputscale, noise, lengthscale)
                )
                label = f"SkyGP-Dense-gPOE(k={k})"
                results[label] = smse
                msll_results[label] = msll
                timing_logs[label] = log
                
                print(f"\n🚀 Training SkyGP-Dense-BCM (k={k})...")
                model = SkyGP_rBCM(
                    x_dim=X_train.shape[1], y_dim=1,
                    max_data_per_expert=max_data_per_expert,
                    max_experts=max_experts,
                    nearest_k=k,
                    replacement=True  # Enable data replacement logic
                )
                smse, msll, log = train_SkyGP(
                    max_samples=max_samples,
                    dataset=dataset,
                    model=model,
                    pretrained_params=(outputscale, noise, lengthscale)
                )
                label = f"SkyGP-Dense-rBCM(k={k})"
                results[label] = smse
                msll_results[label] = msll
                timing_logs[label] = log
                
    return results, msll_results, timing_logs

if __name__ == "__main__":
    results, msll_results, timing_logs = run_all_models(
        model_types=["SkyGP_Fast", "SkyGP_Dense"],
        nearest_k_list=[1,2,4],
        max_samples=50000,
        max_experts=40,
        max_data_per_expert=50,
        dataset="sarcos"  # Change to 'sarcos', 'pumadyn32nm', 'kin40k' or 'electric' as needed
    )
    
    plot_smse_benchmark(results)
    plot_msll_benchmark(msll_results)

    # Print summary of timing logs
    print("\n📊 Summary of Timing (Total and Per Sample):")
    summary = []
    for model_name, log in timing_logs.items():
        num_samples = len(log["samples"])
        total_update_time = np.sum(log["update"])
        total_predict_time = np.sum(log["predict"])

        avg_update_time = total_update_time / num_samples if num_samples > 0 else 0
        avg_predict_time = total_predict_time / num_samples if num_samples > 0 else 0

        summary.append({
            "Model": model_name,
            "Samples": num_samples,
            "Total Update Time (s)": total_update_time,
            "Total Predict Time (s)": total_predict_time,
            "Avg Update Time per Sample (ms)": avg_update_time * 1000,
            "Avg Predict Time per Sample (ms)": avg_predict_time * 1000
        })
    # Convert summary to DataFrame and print
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

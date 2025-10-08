import numpy as np
import time
from tqdm import tqdm
from utils.data_loader import load_data

def train_SkyGP(
    max_samples,
    dataset,
    model,
    pretrained_params
):
    X_train, Y_train, X_test, Y_test = load_data(dataset)
    if dataset in ("kin40k", "twitter"):
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)
    else:
        Y_train = Y_train[:, :1]
        Y_test = Y_test[:, :1]

    outputscale, noise, lengthscale = pretrained_params
    model.init_model_params(0, pretrained_params=(outputscale, noise, lengthscale))

    smse_list = []
    log = {"samples": [], "predict": [], "update": [], "msll": []}

    pbar = tqdm(range(min(max_samples, X_train.shape[0])), desc="Training SkyGP")

    ref_mean = np.mean(Y_train)
    ref_var = np.var(Y_train)
    mse_numerator = 0.0
    msll_raw = []

    for i in pbar:
        x = X_train[i]
        y = Y_train[i]

        t0 = time.time()
        if i > 0:
            y_pred, var = model.predict(x)
            var = np.maximum(var, 1e-3)
        else:
            y_pred = np.zeros_like(y)
            var = np.ones_like(y)
        t1 = time.time()

        t2 = time.time()
        model.add_point(x, y)
        t3 = time.time()

        y_true = y.item()
        if i > 0:
            pred_val = np.atleast_1d(y_pred)[0]
            mse = (y_true - pred_val) ** 2
            mse_numerator += mse
            current_variance = np.var(Y_train[:i + 1])
            smse = mse_numerator / ((i) * current_variance)
        else:
            smse = 1
        smse_list.append(float(smse))

        # --- MSLL likelihood
        _, noise, _ = model.pretrained_params
        sigma_sq = var + noise**2

        if i > 0:
            squared_error = (y_true - y_pred[0])**2
            msll_value = float(0.5 * np.log(2 * np.pi * sigma_sq) + squared_error / (2 * sigma_sq))
            msll_raw.append(msll_value)
        else:   
            msll_raw.append(0.0)

        # Logging
        log["samples"].append(i)
        log["predict"].append(t1 - t0)
        log["update"].append(t3 - t2)

        if (i + 1) % 500 == 0 or i == 0:
            mean_smse = np.mean(smse_list)
            print(f"📊 Step {i+1}: Mean SMSE = {mean_smse:.6f}")

    # --- MSLL post-processing ---
    msll_list = []
    chunk_size = 1000
    N = len(msll_raw)
    msll_raw = np.array(msll_raw)

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        Y_chunk = Y_train[i:end].flatten()
        lik_chunk = msll_raw[i:end]  
        
        baseline_mean = ref_mean
        baseline_var = ref_var
        baseline_term = 0.5 * np.log(2 * np.pi * baseline_var) + \
                        (Y_chunk - baseline_mean) ** 2 / (2 * baseline_var)

        # Adjust MSLL by subtracting the baseline term
        msll_chunk = lik_chunk - baseline_term
        msll_list.extend(msll_chunk.tolist())
    msll_curve = np.cumsum(msll_list) / np.arange(1, len(msll_list) + 1)
    log["msll"] = msll_curve.tolist()
    return smse_list, msll_curve.tolist(), log
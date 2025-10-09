# run_sarcos_with_gpytorch_hparams.py
import os
import numpy as np
from scipy.io import loadmat

from models.skygp import SkyGP_rBCM   # make sure module name matches file case
from utils.gpytorch import fit_hparams_gpytorch


# -----------------------------
# simple config at top
TRAIN_MAT_PATH = "dataset/sarcos/sarcos_inv.mat"
TEST_MAT_PATH  = "dataset/sarcos/sarcos_inv_test.mat"
Y_INDEX        = 0

MAX_DATA_PER_EXPERT = 64
NEAREST_K = 3
MAX_EXPERTS = 16
TIMESCALE = 0.05
REPLACEMENT = False

# gpytorch fitting
GPYTORCH_MAX_POINTS = 1000
GPYTORCH_ITERS = 100
GPYTORCH_LR = 0.1
# -----------------------------


def load_sarcos_mat(train_mat_path, test_mat_path, y_index=0):
    def _read(path):
        md = loadmat(path)
        key = next((k for k in md.keys() if k.startswith("sarcos")), None)
        if key is None:
            raise ValueError(f"Cannot find 'sarcos_*' array in {path}.")
        arr = md[key]
        if arr.ndim != 2 or arr.shape[1] < 28:
            raise ValueError(f"Unexpected shape in {path}: {arr.shape}")
        X = arr[:, :21].astype(np.float64)
        Y = arr[:, 21:].astype(np.float64)
        return X, Y
    Xtr, Ytr = _read(train_mat_path)
    Xte, Yte = _read(test_mat_path)
    Ytr = Ytr[:, y_index:y_index+1]
    Yte = Yte[:, y_index:y_index+1]
    return Xtr, Ytr, Xte, Yte


def standardize_train_apply(X_train, Y_train, X_test, Y_test):
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_std[X_std < 1e-9] = 1.0
    Xtr = (X_train - X_mean) / X_std
    Xte = (X_test  - X_mean) / X_std

    Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0)
    Y_std[Y_std < 1e-9] = 1.0
    Ytr = (Y_train - Y_mean) / Y_std
    Yte = (Y_test  - Y_mean) / Y_std

    return Xtr.astype(np.float32), Ytr.astype(np.float32), Xte.astype(np.float32), Yte.astype(np.float32)


def main():
    if not os.path.isfile(TRAIN_MAT_PATH) or not os.path.isfile(TEST_MAT_PATH):
        raise FileNotFoundError("Put sarcos_inv.mat & sarcos_inv_test.mat in dataset/sarcos/.")
    X_train, Y_train, X_test, Y_test = load_sarcos_mat(TRAIN_MAT_PATH, TEST_MAT_PATH, Y_INDEX)
    Xtr, Ytr, Xte, Yte = standardize_train_apply(X_train, Y_train, X_test, Y_test)
    Ntr, D = Xtr.shape
    Nte = Xte.shape[0]
    print(f"Train {Ntr}x{D}, Test {Nte}x{D}, y-dim=1 (output #{Y_INDEX})")

    # === (A) learn hyperparameters with GPyTorch on a subset of TRAIN ===
    print("🔧 Learning hyperparameters via GPyTorch ...")
    outputscale, noise, lengthscale = fit_hparams_gpytorch(
        Xtr, Ytr[:, 0],
        max_points=GPYTORCH_MAX_POINTS,
        iters=GPYTORCH_ITERS,
        lr=GPYTORCH_LR,
        use_cuda_if_available=True,
        print_every=50,
    )
    print("[HP] sigma_f:", outputscale, "noise:", noise, "lengthscale[:5]:", lengthscale[:5])

    # === (B) build SkyGP with learned hparams ===
    model = SkyGP_rBCM(
        x_dim=D,
        y_dim=1,
        max_data_per_expert=MAX_DATA_PER_EXPERT,
        nearest_k=NEAREST_K,
        max_experts=MAX_EXPERTS,
        replacement=REPLACEMENT,
        pretrained_params=(outputscale, noise, lengthscale),
        timescale=TIMESCALE,
    )

    # Offline pretrain to populate experts (now that hparams are set, we disable internal optimization)
    print("🏋️ Offline pretraining (populate experts with learned hparams) ...")
    model.offline_pretrain(
        Xtr, Ytr,
        max_samples=None,
        show_progress=True,
        optimize_hparams=False,  # we already fitted with GPyTorch
    )

    # === (C) Online: predict-then-update on TEST stream ===
    print("🔮 Online prediction + update ...")
    y_var_train = float(Ytr.var()) if Ytr.var() > 0 else 1.0
    mse_sum = 0.0
    msll_sum = 0.0

    # noise for MSLL (std, already extracted)
    noise_scalar = float(noise[0])
    report_every = max(1, Nte // 10)

    for i in range(Nte):
        x = Xte[i]
        y = Yte[i]
        y_pred, var = model.online_step(x, y)  # predict first, then update

        err = float(y[0] - y_pred[0])
        mse_sum += err * err
        smse = mse_sum / ((i + 1) * y_var_train)

        sigma_sq = float(var[0]) + noise_scalar**2
        msll_inst = 0.5 * np.log(2 * np.pi * sigma_sq) + (err * err) / (2 * sigma_sq)
        msll_sum += msll_inst
        msll_avg = msll_sum / (i + 1)

        if (i + 1) % report_every == 0 or i == 0:
            print(f"[{i+1}/{Nte}] SMSE={smse:.4f} | MSLL={msll_avg:.4f}")

    print("\n✅ Done.")
    print(f"Final SMSE={smse:.6f} | MSLL={msll_avg:.6f}")


if __name__ == "__main__":
    main()

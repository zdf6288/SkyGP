# gpytorch_hparam_fit.py
import numpy as np
import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@torch.no_grad()
def _to_numpy_hparams(model, likelihood):
    # outputscale
    sf2 = model.covar_module.outputscale.detach().cpu().item()
    sf = np.sqrt(sf2) if sf2 > 0 else 1.0  # SkyGP expects sigma_f (std), not variance
    # noise (std)
    sn2 = likelihood.noise.detach().cpu().item()
    sn = np.sqrt(sn2) if sn2 > 0 else 1e-3
    # ARD lengthscales (gpytorch stores lengthscale^2 under the hood; attribute is already ls)
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().view(-1).numpy()
    return np.array([sf], dtype=float), np.array([sn], dtype=float), ls.astype(float)


def fit_hparams_gpytorch(
    X, y,
    max_points=4000,
    iters=300,
    lr=0.1,
    use_cuda_if_available=True,
    print_every=50,
):
    """
    Fit ARD RBF hyperparameters with GPyTorch Exact GP.

    Args
    ----
    X : (N, D) float64/32
    y : (N,) or (N,1)
    max_points : number of points for fitting (subsample if larger)
    iters : training iterations
    lr : Adam learning rate

    Returns
    -------
    (outputscale, noise, lengthscale) as numpy arrays:
      outputscale: shape (1,)  -> sigma_f
      noise:       shape (1,)  -> sigma_n
      lengthscale: shape (D,)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.float32)
    N, D = X.shape

    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    device = "cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu"
    tx = torch.from_numpy(Xs).to(device)
    ty = torch.from_numpy(ys).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(tx, ty, likelihood, ard_num_dims=D).to(device)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(1, iters + 1):
        optimizer.zero_grad()
        output = model(tx)
        loss = -mll(output, ty)
        loss.backward()
        optimizer.step()
        if print_every and (i % print_every == 0 or i == 1 or i == iters):
            sf2 = model.covar_module.outputscale.detach().item()
            sn2 = likelihood.noise.detach().item()
            print(f"[GPyTorch opt] iter {i:4d} | nll {loss.item():.4f} "
                  f"| sf^2 {sf2:.4g} | sn^2 {sn2:.4g}")

    model.eval(); likelihood.eval()
    return _to_numpy_hparams(model, likelihood)
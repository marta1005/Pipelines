import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def MultiOutputGradientBoosting(**kwargs):
    """GradientBoostingRegressor wrapped for multioutput regression."""
    return MultiOutputRegressor(GradientBoostingRegressor(**kwargs))


def RandomForest(**kwargs):
    """RandomForestRegressor — natively multi-output, no wrapper needed."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**kwargs)


def MultiOutputXGBoost(**kwargs):
    """XGBRegressor wrapped for multioutput regression.

    XGBoost is single-output natively; MultiOutputRegressor trains one
    independent model per output column.
    """
    from xgboost import XGBRegressor
    return MultiOutputRegressor(XGBRegressor(**kwargs))


try:
    import torch.nn as _nn

    class _ResBlock(_nn.Module):
        """Residual block: Linear→LayerNorm→GELU→Linear→LayerNorm + skip + GELU."""
        def __init__(self, d):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(d, d),
                _nn.LayerNorm(d),
                _nn.GELU(),
                _nn.Linear(d, d),
                _nn.LayerNorm(d),
            )
            self.act = _nn.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))

except ImportError:
    _ResBlock = None


class PyTorchNN(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for a deep residual PyTorch network.

    Architecture: fully connected residual blocks (2 linear layers per block
    + skip connection), LayerNorm, GELU activation.  Substantially deeper and
    more expressive than sklearn MLPRegressor while fitting inside the SF
    sklearn pipeline.

    Parameters
    ----------
    n_hidden : int
        Width of each hidden layer.
    n_layers : int
        Number of residual blocks (each block = 2 linear layers).
    lr : float
        Initial learning rate for Adam.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience (epochs without val loss improvement).
    weight_decay : float
        L2 regularisation strength.
    loss : str
        'mse', 'mae', or 'huber'.
    random_state : int or None
        Seed for reproducibility.
    verbose : bool
        Print training progress every 50 epochs.
    """

    def __init__(
        self,
        n_hidden=128,
        n_layers=4,
        lr=1e-3,
        epochs=500,
        batch_size=64,
        patience=30,
        weight_decay=1e-4,
        loss='mse',
        random_state=42,
        verbose=True,
    ):
        self.n_hidden  = n_hidden
        self.n_layers  = n_layers
        self.lr        = lr
        self.epochs    = epochs
        self.batch_size = batch_size
        self.patience  = patience
        self.weight_decay = weight_decay
        self.loss      = loss
        self.random_state = random_state
        self.verbose   = verbose

    # ── internal helpers ────────────────────────────────────────────────────────

    def _build_network(self, n_in, n_out):
        import torch.nn as nn

        layers = [nn.Linear(n_in, self.n_hidden), nn.LayerNorm(self.n_hidden), nn.GELU()]
        for _ in range(self.n_layers):
            layers.append(_ResBlock(self.n_hidden))
        layers.append(nn.Linear(self.n_hidden, n_out))
        return nn.Sequential(*layers)

    # ── sklearn API ─────────────────────────────────────────────────────────────

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        self.n_outputs_ = y_np.shape[1]

        # Standardise Y so all outputs contribute equally to MSE
        self.y_mean_ = y_np.mean(axis=0)
        self.y_std_  = y_np.std(axis=0)
        self.y_std_[self.y_std_ == 0] = 1.0   # avoid division by zero
        y_np = (y_np - self.y_mean_) / self.y_std_

        # train/val split (15 % val for early stopping)
        n_val = max(1, int(0.15 * len(X_np)))
        idx = np.random.permutation(len(X_np))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        X_tr = torch.tensor(X_np[tr_idx]).to(device)
        y_tr = torch.tensor(y_np[tr_idx]).to(device)
        X_vl = torch.tensor(X_np[val_idx]).to(device)
        y_vl = torch.tensor(y_np[val_idx]).to(device)

        net = self._build_network(X_np.shape[1], self.n_outputs_).to(device)
        opt = Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

        loss_fn = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.HuberLoss()}[self.loss]

        best_val, patience_cnt = float('inf'), 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(len(X_tr), device=device)
            for i in range(0, len(X_tr), self.batch_size):
                idx_b = perm[i: i + self.batch_size]
                opt.zero_grad()
                loss_fn(net(X_tr[idx_b]), y_tr[idx_b]).backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_vl), y_vl).item()
            sched.step(val_loss)

            if val_loss < best_val - 1e-8:
                best_val, patience_cnt = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    if self.verbose:
                        print(f'  [PyTorchNN] Early stop @ epoch {epoch}  (best val loss {best_val:.6f})')
                    break

            if self.verbose and epoch % 50 == 0:
                print(f'  [PyTorchNN] epoch {epoch:4d}  val_loss {val_loss:.6f}  lr {opt.param_groups[0]["lr"]:.2e}')

        if best_state is not None:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        self.net_  = net.cpu()
        self.device_ = 'cpu'
        return self

    def predict(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32))
            out = self.net_(X_t).numpy()
        out = out * self.y_std_ + self.y_mean_
        return out if out.shape[1] > 1 else out.ravel()

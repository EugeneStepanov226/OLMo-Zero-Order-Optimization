"""First-order Muon optimizer for OLMo training.

Muon (Momentum + Orthogonalization via Newton-Schulz) is a first-order optimizer
that uses standard backprop gradients and applies Newton-Schulz orthogonalization
to the gradient matrix for 2-D parameters, producing unit-singular-value updates.

For 1-D parameters (biases, norms) standard SGD with Nesterov momentum is used.

Reference: Keller Jordan, "Muon: An optimizer for hidden layers", 2024.
           https://github.com/KellerJordan/modded-nanogpt
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer


def _newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix sign function via Newton-Schulz iteration.

    Maps G → U Vᵀ (where G = U Σ Vᵀ), i.e. normalizes all singular values to ~1.
    Always operates in float32.
    """
    assert G.ndim == 2, "G must be a 2-D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class FOMuon(Optimizer):
    """First-order Muon optimizer.

    For each 2-D weight matrix W:
      1. Accumulates Nesterov momentum:  buf = momentum * buf + grad
         effective gradient:             g   = grad + momentum * buf  (nesterov)
      2. Applies Newton-Schulz:          update = NS(g)   (singular values → 1)
      3. Decoupled weight decay:         W *= (1 - lr * weight_decay)
      4. Update:                         W -= lr * update

    For 1-D parameters (biases, LayerNorm weights, norms):
      Standard SGD with Nesterov momentum (no orthogonalization).

    DDP: gradients are already all-reduced by DDP before step() is called,
    so no special inter-rank logic is needed.

    FSDP: not supported (weights must be full tensors for NS to be meaningful).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            max_grad_norm = group["max_grad_norm"]

            # Optional per-group gradient clipping (Frobenius norm across all 2-D params).
            if max_grad_norm is not None:
                params_2d = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
                if params_2d:
                    total_norm = torch.linalg.vector_norm(
                        torch.stack([p.grad.norm() for p in params_2d])
                    )
                    clip_coef = max_grad_norm / (total_norm + 1e-6)
                    if clip_coef < 1.0:
                        for p in params_2d:
                            p.grad.mul_(clip_coef)

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()  # work in fp32

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    effective_g = g.add(buf, alpha=momentum)
                else:
                    effective_g = buf

                # Decoupled weight decay.
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                if p.ndim == 2:
                    update = _newtonschulz5(effective_g, steps=ns_steps)
                    p.data.add_(update.to(p.dtype), alpha=-lr)
                else:
                    p.data.add_(effective_g.to(p.dtype), alpha=-lr)

        return loss

"""
HybridZOMuon: ZOMuon для тела модели (2D веса) + AdamW для головы (FO, с backward).

Схема одного шага:
  1. ZO-фаза  – ZOMuon по zo_params  (2·q forward-проходов, без backward)
  2. FO-фаза  – AdamW  по fo_params  (1 forward + 1 backward)

Разделение параметров:
  fo_params  – параметры, имена которых содержат хотя бы одну строку из fo_param_patterns
               (по умолчанию ["ff_out"] – выходная голова LM)
  zo_params  – всё остальное

DDP-корректность:
  - ZO-параметры: requires_grad=False → DDP их не трогает
  - FO-параметры: requires_grad=True  → DDP синхронизирует через backward()
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from .zo_optim import ZeroOrderOptimizer, ZOMuon, _newtonschulz5


class HybridZOMuon(ZeroOrderOptimizer):
    """Гибридный оптимизатор: ZOMuon (ZO) для тела + AdamW (FO) для головы."""

    def __init__(
        self,
        zo_params: List[dict],          # param_groups для ZOMuon
        fo_params: List[dict],          # param_groups для AdamW
        # ZO гиперпараметры
        lr: float = 7e-4,
        zo_eps: float = 1e-3,
        rank: int = 64,
        step_interval: int = 100,
        num_samples: int = 1,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        max_grad_norm: Optional[float] = None,
        # FO гиперпараметры
        fo_lr: float = 1e-4,
        fo_betas: tuple = (0.9, 0.95),
        fo_eps: float = 1e-8,
        fo_weight_decay: float = 0.1,
        fo_max_grad_norm: Optional[float] = 1.0,
    ):
        # Помечаем группы, чтобы step() знал что делать с каждой
        for g in zo_params:
            g["_hybrid_role"] = "zo"
            g["_base_lr"] = lr        # базовый LR для scheduler'а
            g.setdefault("zo_eps", zo_eps)
            g.setdefault("weight_decay", weight_decay)
            g.setdefault("lr", lr)
        for g in fo_params:
            g["_hybrid_role"] = "fo"
            g["_base_lr"] = fo_lr     # отдельный базовый LR для FO
            g.setdefault("weight_decay", fo_weight_decay)
            g.setdefault("lr", fo_lr)

        super().__init__(zo_params + fo_params, defaults={})

        # --- Внутренний ZOMuon (только ZO-группы) ---
        self._zo = ZOMuon(
            zo_params,
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            num_samples=num_samples,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )

        # --- FO гиперпараметры ---
        self._fo_lr = fo_lr
        self._fo_betas = fo_betas
        self._fo_eps = fo_eps
        self._fo_weight_decay = fo_weight_decay
        self._fo_max_grad_norm = fo_max_grad_norm
        self._fo_step_count = 0

        # Убедимся что FO-параметры требуют градиент
        for g in fo_params:
            for p in g["params"]:
                p.requires_grad_(True)

        self._last_metrics: dict = {}

    # ------------------------------------------------------------------
    # Вспомогательные итераторы
    # ------------------------------------------------------------------

    def _fo_param_groups(self):
        return [g for g in self.param_groups if g.get("_hybrid_role") == "fo"]

    def _zo_param_groups(self):
        return [g for g in self.param_groups if g.get("_hybrid_role") == "zo"]

    # ------------------------------------------------------------------
    # Синхронизация LR из scheduler'а с внутренним ZOMuon
    # ------------------------------------------------------------------

    def _sync_zo_lr(self):
        """Копирует lr из внешних zo-групп во внутренний _zo."""
        for ext_g, int_g in zip(self._zo_param_groups(), self._zo.param_groups):
            int_g["lr"] = ext_g["lr"]

    # ------------------------------------------------------------------
    # FO: AdamW update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_fo_adamw(self):
        """AdamW шаг для FO-параметров (градиенты уже вычислены)."""
        self._fo_step_count += 1
        b1, b2 = self._fo_betas
        bias_c1 = 1.0 - b1 ** self._fo_step_count
        bias_c2 = 1.0 - b2 ** self._fo_step_count

        update_norm_sq = 0.0
        for group in self._fo_param_groups():
            lr = group["lr"]
            wd = group.get("weight_decay", self._fo_weight_decay)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(b1).add_(g, alpha=1.0 - b1)
                exp_avg_sq.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                # Bias-corrected update
                step_size = lr / bias_c1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(self._fo_eps)
                update = exp_avg / denom
                update_norm_sq += update.norm().item() ** 2

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(update.to(p.dtype), alpha=-step_size)

        self._last_metrics["fo_update_norm"] = math.sqrt(update_norm_sq)

    # ------------------------------------------------------------------
    # Основной шаг
    # ------------------------------------------------------------------

    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
        fo_closure: Optional[Callable[[], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        closure     – ZO-closure (inference_mode, без grad) → скалярный лосс
        fo_closure  – FO-closure (с enable_grad)           → лосс tensor с grad
                      Если None – FO шаг пропускается.
        """
        # --- ZO шаг ---
        self._sync_zo_lr()
        loss = self._zo.step(closure, z_seed=z_seed)

        # Копируем ZO-метрики
        for k, v in self._zo.get_post_step_metrics().items():
            self._last_metrics[k] = v.item()

        # --- FO шаг ---
        if fo_closure is not None:
            # Вычисляем градиенты для FO-параметров
            fo_loss = fo_closure()  # forward с enable_grad

            # Опциональный gradient clipping для FO
            if self._fo_max_grad_norm is not None:
                fo_params = [p for g in self._fo_param_groups() for p in g["params"] if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(fo_params, self._fo_max_grad_norm)

            fo_grad_norm = math.sqrt(
                sum(p.grad.float().norm().item() ** 2
                    for g in self._fo_param_groups()
                    for p in g["params"] if p.grad is not None)
            )
            self._last_metrics["fo_grad_norm"] = fo_grad_norm

            self._apply_fo_adamw()

            # Обнуляем градиенты FO-параметров
            for group in self._fo_param_groups():
                for p in group["params"]:
                    p.grad = None

        return loss

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}


# ------------------------------------------------------------------
# Утилита: разбить параметры модели на ZO и FO группы
# ------------------------------------------------------------------

def split_params_fo_zo(
    model: nn.Module,
    fo_param_patterns: List[str],
    weight_decay: float,
    decay_norm_and_bias: bool,
    decay_embeddings: bool,
) -> tuple[List[dict], List[dict]]:
    """
    Разбивает параметры модели на:
      fo_groups – параметры, имена которых содержат хотя бы одну строку из fo_param_patterns
      zo_groups – всё остальное

    Внутри каждой группы отдельно выделяются decay / no-decay параметры.
    """
    fo_decay, fo_no_decay = [], []
    zo_decay, zo_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_fo = any(pat in name for pat in fo_param_patterns)

        # Определяем weight-decay
        is_1d = param.ndim == 1
        is_embedding = "wte" in name or "embedding" in name.lower()
        apply_wd = weight_decay != 0.0 and not (
            (is_1d and not decay_norm_and_bias)
            or (is_embedding and not decay_embeddings)
        )

        if is_fo:
            (fo_decay if apply_wd else fo_no_decay).append(param)
        else:
            (zo_decay if apply_wd else zo_no_decay).append(param)

    fo_groups = []
    if fo_decay:
        fo_groups.append({"params": fo_decay, "weight_decay": weight_decay})
    if fo_no_decay:
        fo_groups.append({"params": fo_no_decay, "weight_decay": 0.0})

    zo_groups = []
    if zo_decay:
        zo_groups.append({"params": zo_decay, "weight_decay": weight_decay})
    if zo_no_decay:
        zo_groups.append({"params": zo_no_decay, "weight_decay": 0.0})

    return zo_groups, fo_groups

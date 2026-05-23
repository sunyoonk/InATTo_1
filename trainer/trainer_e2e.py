"""End-to-end trainer for InATTo (spec §6).

Single optimizer with two parameter groups (tokenizer/alignment vs T5),
phase-aware freezing via the model's ``set_phase`` hook.

Logging per epoch: all loss components, phi histogram bins, codebook
utilization, identifier change rate (vs previous epoch).
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _safe_mean(xs):
    return float(np.mean(xs)) if xs else 0.0


class E2ETrainer:
    def __init__(
        self,
        model,
        schedule,
        lr_tokenizer: float = 1e-3,
        lr_t5: float = 1e-4,
        weight_decay: float = 0.0,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.schedule = schedule
        self.device = torch.device(device)
        self.model.to(self.device)

        # Two param groups: tokenizer/alignment vs T5.
        tok_params = [p for p in self.model.tokenizer.parameters() if p.requires_grad or True]
        algn_params = (list(self.model.align_user.parameters())
                        + list(self.model.align_item.parameters()))
        t5_params = list(self.model.bridge.t5.parameters())
        # Note: descriptor MiniLM is frozen always.
        self.optim = torch.optim.AdamW([
            {"params": tok_params + algn_params, "lr": lr_tokenizer,
             "weight_decay": weight_decay},
            {"params": t5_params, "lr": lr_t5,
             "weight_decay": weight_decay},
        ])

        self._prev_codes: dict[str, torch.Tensor] = {}

    # ----------------------------------------------------------------------
    def _phi_hist(self, phi: torch.Tensor, bins: int = 10) -> list[float]:
        h, _ = np.histogram(phi.detach().cpu().numpy().flatten(),
                            bins=bins, range=(0, 1))
        return (h / h.sum()).tolist() if h.sum() > 0 else h.tolist()

    def _codebook_util(self, codes: torch.Tensor, V: int) -> float:
        unique = torch.unique(codes).numel()
        return unique / V

    def _id_change_rate(self, key: str, codes: torch.Tensor) -> float | None:
        if key not in self._prev_codes:
            self._prev_codes[key] = codes.detach().cpu()
            return None
        prev = self._prev_codes[key]
        # Align lengths if needed (eval set may grow if dataset shifts; assume same)
        n = min(prev.shape[0], codes.shape[0])
        diff = (prev[:n] != codes.detach().cpu()[:n]).any(dim=(1, 2))
        rate = diff.float().mean().item()
        self._prev_codes[key] = codes.detach().cpu()
        return rate

    # ----------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader, phase: str) -> dict:
        self.model.train()
        self.model.set_phase(phase)
        agg = {k: [] for k in ("L_total", "L_gen", "L_recon", "L_Q",
                                "L_align", "L_align_u", "L_align_i",
                                "L_ui", "L_rate")}
        phi_user_all = []
        phi_item_all = []
        last_user_codes = None
        last_item_codes = None

        for uids, tgts, hist, valid in loader:
            uids = uids.to(self.device, non_blocking=True)
            tgts = tgts.to(self.device, non_blocking=True)
            hist = hist.to(self.device, non_blocking=True)
            valid = valid.to(self.device, non_blocking=True)

            out = self.model(uids, tgts, hist, valid, phase=phase)
            loss = out["L_total"]
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            for k in agg:
                agg[k].append(out[k].item())
            phi_user_all.append(out["phi_user"].detach())
            phi_item_all.append(out["phi_item"].detach())
            last_user_codes = out["user_codes"].detach()
            last_item_codes = out["item_codes"].detach()

        phi_u = torch.cat(phi_user_all, dim=0) if phi_user_all else None
        phi_i = torch.cat(phi_item_all, dim=0) if phi_item_all else None

        return {
            "phase": phase,
            **{k: _safe_mean(v) for k, v in agg.items()},
            "phi_user_mean": float(phi_u.mean().item()) if phi_u is not None else None,
            "phi_item_mean": float(phi_i.mean().item()) if phi_i is not None else None,
            "phi_user_hist": self._phi_hist(phi_u) if phi_u is not None else [],
            "phi_item_hist": self._phi_hist(phi_i) if phi_i is not None else [],
            "codebook_util_item": (
                self._codebook_util(last_item_codes, self.model.codebook.V)
                if last_item_codes is not None else None
            ),
        }

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def quick_eval(self, loader: DataLoader, phase: str = "joint",
                   max_batches: int = 50) -> dict:
        """Run a few batches in eval mode; mean L_gen as a quick proxy."""
        self.model.eval()
        self.model.set_phase(phase)
        L_gen_list = []
        for i, (uids, tgts, hist, valid) in enumerate(loader):
            if i >= max_batches:
                break
            uids = uids.to(self.device); tgts = tgts.to(self.device)
            hist = hist.to(self.device); valid = valid.to(self.device)
            out = self.model(uids, tgts, hist, valid, phase=phase)
            L_gen_list.append(out["L_gen"].item())
        return {"val_L_gen": _safe_mean(L_gen_list)}

    # ----------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        log_path: str | Path | None = None,
        ckpt_save_path: str | Path | None = None,
        save_every: int = 1,
        resume_from: str | Path | None = None,
    ) -> list[dict]:
        """Train loop with per-epoch checkpointing and resume.

        Parameters
        ----------
        ckpt_save_path : path or None
            If provided, model state + optimizer state + history are dumped
            here every `save_every` epochs (overwrites the previous snapshot).
            Use this to survive crashes / OOM during long runs.
        save_every : int
            Save frequency in epochs. Default 1 (every epoch).
        resume_from : path or None
            If provided and exists, load model + optimizer + history before
            starting; training continues from the next epoch.
        """
        history: list[dict] = []
        start_epoch = 0

        if resume_from is not None and Path(resume_from).exists():
            print(f"[resume] loading {resume_from}")
            ck = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model_state"])
            if "optim_state" in ck:
                self.optim.load_state_dict(ck["optim_state"])
            history = ck.get("history", [])
            start_epoch = ck.get("next_epoch", len(history))
            print(f"[resume] continuing from epoch {start_epoch}  (history={len(history)} epochs)")

        for epoch in range(start_epoch, self.schedule.total_epochs):
            phase = self.schedule.phase_of(epoch)
            t0 = time.time()
            train_log = self.train_epoch(train_loader, phase)
            log = {"epoch": epoch, "phase": phase, "epoch_time": time.time() - t0,
                   **train_log}

            # Identifier change rate (item-side, on a fixed mini-batch for stability)
            # We just use last training batch's codes here.

            if val_loader is not None and (epoch + 1) % 1 == 0:
                log.update(self.quick_eval(val_loader, phase=phase))

            print(
                f"[ep {epoch:>2}/{self.schedule.total_epochs} | {phase}] "
                f"L={log['L_total']:.3f}  gen={log['L_gen']:.3f}  "
                f"recon={log['L_recon']:.3f}  Q={log['L_Q']:.3f}  "
                f"align={log['L_align']:.3f}  ui={log['L_ui']:.3f}  "
                f"rate={log['L_rate']:.3f}  "
                f"phi_u={log['phi_user_mean']:.3f}  phi_i={log['phi_item_mean']:.3f}  "
                f"cb_util_i={log['codebook_util_item']:.3f}  "
                f"({log['epoch_time']:.1f}s)"
            )
            history.append(log)
            if log_path is not None:
                Path(log_path).write_text(json.dumps(history, indent=2))

            # Per-epoch checkpoint for OOM/crash recovery.
            if ckpt_save_path is not None and (epoch + 1) % save_every == 0:
                Path(ckpt_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optim.state_dict(),
                    "history": history,
                    "next_epoch": epoch + 1,
                }, ckpt_save_path)

        return history

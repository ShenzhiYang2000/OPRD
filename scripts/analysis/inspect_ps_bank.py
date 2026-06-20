#!/usr/bin/env python3
"""Inspect pre-experiment 2 ps_bank.pt: verify P_S and P_T were saved."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ps_bank.pt checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to rank_*/ps_bank.pt")
    return parser.parse_args()


def tensor_norm(t: torch.Tensor) -> float:
    return float(t.detach().float().norm().item())


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: {path}")
    print(f"rank: {ckpt.get('rank')}")
    print(f"best_epoch: {ckpt.get('best_epoch', 'n/a')}")
    print(f"saved_from: {ckpt.get('saved_from', 'unknown')}")
    print(f"subspace_mode: {ckpt.get('subspace_mode', 'full')}")
    print(f"projector_type: {ckpt.get('projector_type', 'linear')}")
    if ckpt.get("projector_type") == "mlp":
        print(f"mlp_hidden_mult: {ckpt.get('mlp_hidden_mult')}")
    print(f"num layer_pairs: {len(ckpt.get('layer_pairs', []))}")

    state_dict = ckpt.get("state_dict", {})
    ps_keys = sorted(k for k in state_dict if k.startswith("projectors.") and k.endswith(".weight"))
    print(f"\nP_S weight keys in state_dict: {len(ps_keys)}")
    for key in ps_keys[:3]:
        print(f"  {key}: shape={tuple(state_dict[key].shape)} norm={tensor_norm(state_dict[key]):.4f}")
    if len(ps_keys) > 3:
        print(f"  ... ({len(ps_keys) - 3} more)")

    frozen_w = ckpt.get("frozen_pt_weights", {})
    frozen_m = ckpt.get("frozen_pt_means", {})
    pt_state = ckpt.get("pt_state_dict", {})
    print(f"\nfrozen_pt_weights keys: {len(frozen_w)}")
    if ckpt.get("subspace_mode") == "direct":
        pt_keys = sorted(k for k in pt_state if k.startswith("projectors.") and k.endswith(".weight"))
        print(f"trainable P_T keys in pt_state_dict: {len(pt_keys)}")
        for key in pt_keys[:3]:
            print(f"  {key}: shape={tuple(pt_state[key].shape)} norm={tensor_norm(pt_state[key]):.4f}")
        if len(pt_keys) > 3:
            print(f"  ... ({len(pt_keys) - 3} more)")
    elif not frozen_w:
        print("  WARNING: frozen_pt_weights is empty — P_T was NOT saved in this file.")
    else:
        for key in sorted(frozen_w.keys())[:3]:
            w = frozen_w[key]
            print(f"  {key}: shape={tuple(w.shape)} norm={tensor_norm(w):.4f}")
        if len(frozen_w) > 3:
            print(f"  ... ({len(frozen_w) - 3} more)")

    print(f"\nfrozen_pt_means keys: {len(frozen_m)}")
    if frozen_m:
        sample_key = sorted(frozen_m.keys())[0]
        print(f"  example {sample_key}: norm={tensor_norm(frozen_m[sample_key]):.4f}")

    if frozen_w:
        sample_keys = list(frozen_w.keys())[:5]
        has_legacy = any("_" in k and not k.startswith("s") for k in sample_keys)
        has_new = any(k.startswith("s") and "_t" in k for k in sample_keys)
        print(f"\nKey format: legacy(0_0)={has_legacy or any(k[0].isdigit() for k in sample_keys)} new(s0_t0)={has_new}")


if __name__ == "__main__":
    main()

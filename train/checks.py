from typing import Dict, Optional
import csv
import os
import torch


def _max_abs_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    denom = a.abs().clamp_min(eps)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


@torch.no_grad()
def check_transform_inverse_identity(
    encoderT,
    loader,
    device: str,
    max_batches: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    log_csv_path: Optional[str] = None,
) -> Dict:
    """
    TWO checks per scale (b,s3,s4):

    (1) Round-trip on ORIGINAL:
        z  = forward_original(x)
        z' = T(z)
        z_rt = inverse(z')
        check z ~= z_rt

    (2) Federated-path recovery:
        z  = forward_original(x)
        z'_net = forward(x)         # what client sends
        z_srv  = inverse(z'_net)    # what server recovers using same client's params
        check z ~= z_srv

    This matches your federated assumption: server only sees protected latents.
    """
    encoderT.eval().to(device)

    stats = {
        "roundtrip_max_abs": {"b": 0.0, "s3": 0.0, "s4": 0.0},
        "roundtrip_max_rel": {"b": 0.0, "s3": 0.0, "s4": 0.0},
        "federated_max_abs": {"b": 0.0, "s3": 0.0, "s4": 0.0},
        "federated_max_rel": {"b": 0.0, "s3": 0.0, "s4": 0.0},
    }

    nb = 0
    for x, _, _ in loader:
        x = x.to(device)
        nb += 1

        # ORIGINAL (unprotected)
        b, s3, s4 = encoderT.forward_original(x)

        if getattr(encoderT, "use_klt", False):
            # (1) Round-trip: inverse(T(z))
            b_rt  = encoderT.T_b.inverse(encoderT.T_b(b))
            s3_rt = encoderT.T_s3.inverse(encoderT.T_s3(s3))
            s4_rt = encoderT.T_s4.inverse(encoderT.T_s4(s4))

            # (2) Federated path: inverse(encoderT(x)) where encoderT(x) returns protected
            bT_net, s3T_net, s4T_net = encoderT(x)
            b_srv, s3_srv, s4_srv = encoderT.inverse_triplet(bT_net, s3T_net, s4T_net)
        else:
            b_rt = b_srv = b
            s3_rt = s3_srv = s3
            s4_rt = s4_srv = s4

        for k, z, z_rt, z_srv in [
            ("b", b, b_rt, b_srv),
            ("s3", s3, s3_rt, s3_srv),
            ("s4", s4, s4_rt, s4_srv),
        ]:
            ma, mr = _max_abs_rel(z, z_rt)
            stats["roundtrip_max_abs"][k] = max(stats["roundtrip_max_abs"][k], ma)
            stats["roundtrip_max_rel"][k] = max(stats["roundtrip_max_rel"][k], mr)

            ma2, mr2 = _max_abs_rel(z, z_srv)
            stats["federated_max_abs"][k] = max(stats["federated_max_abs"][k], ma2)
            stats["federated_max_rel"][k] = max(stats["federated_max_rel"][k], mr2)

        if nb >= int(max_batches):
            break

    def _ok(ma, mr):
        return (ma <= atol) or (mr <= rtol)

    ok = {
        "roundtrip": {k: _ok(stats["roundtrip_max_abs"][k], stats["roundtrip_max_rel"][k]) for k in ["b", "s3", "s4"]},
        "federated": {k: _ok(stats["federated_max_abs"][k], stats["federated_max_rel"][k]) for k in ["b", "s3", "s4"]},
    }

    print("[Transform Check] z ~= inverse(T(z))  (round-trip)")
    for k in ["b", "s3", "s4"]:
        print(f"  {k:>2}: max_abs={stats['roundtrip_max_abs'][k]:.3e}  max_rel={stats['roundtrip_max_rel'][k]:.3e}  OK={ok['roundtrip'][k]}")

    print("[Transform Check] z ~= inverse(encoderT(x))  (federated-path)")
    for k in ["b", "s3", "s4"]:
        print(f"  {k:>2}: max_abs={stats['federated_max_abs'][k]:.3e}  max_rel={stats['federated_max_rel'][k]:.3e}  OK={ok['federated'][k]}")

    # optional CSV log (single row)
    if log_csv_path is not None:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        fields = [
            "roundtrip_b_max_abs","roundtrip_b_max_rel","roundtrip_s3_max_abs","roundtrip_s3_max_rel","roundtrip_s4_max_abs","roundtrip_s4_max_rel",
            "federated_b_max_abs","federated_b_max_rel","federated_s3_max_abs","federated_s3_max_rel","federated_s4_max_abs","federated_s4_max_rel",
            "roundtrip_b_ok","roundtrip_s3_ok","roundtrip_s4_ok",
            "federated_b_ok","federated_s3_ok","federated_s4_ok",
            "atol","rtol","max_batches"
        ]
        row = {
            "roundtrip_b_max_abs": stats["roundtrip_max_abs"]["b"],
            "roundtrip_b_max_rel": stats["roundtrip_max_rel"]["b"],
            "roundtrip_s3_max_abs": stats["roundtrip_max_abs"]["s3"],
            "roundtrip_s3_max_rel": stats["roundtrip_max_rel"]["s3"],
            "roundtrip_s4_max_abs": stats["roundtrip_max_abs"]["s4"],
            "roundtrip_s4_max_rel": stats["roundtrip_max_rel"]["s4"],
            "federated_b_max_abs": stats["federated_max_abs"]["b"],
            "federated_b_max_rel": stats["federated_max_rel"]["b"],
            "federated_s3_max_abs": stats["federated_max_abs"]["s3"],
            "federated_s3_max_rel": stats["federated_max_rel"]["s3"],
            "federated_s4_max_abs": stats["federated_max_abs"]["s4"],
            "federated_s4_max_rel": stats["federated_max_rel"]["s4"],
            "roundtrip_b_ok": ok["roundtrip"]["b"],
            "roundtrip_s3_ok": ok["roundtrip"]["s3"],
            "roundtrip_s4_ok": ok["roundtrip"]["s4"],
            "federated_b_ok": ok["federated"]["b"],
            "federated_s3_ok": ok["federated"]["s3"],
            "federated_s4_ok": ok["federated"]["s4"],
            "atol": atol,
            "rtol": rtol,
            "max_batches": max_batches,
        }
        with open(log_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerow(row)

    return {"stats": stats, "ok": ok}


@torch.no_grad()
def extract_latents_server_inverse(encoderT, loader, device: str):
    """
    Federated-consistent extraction for UMN training:

      Client: bT,s3T,s4T = encoderT(x)           # protected sent over network
      Server: b,s3,s4 = inverse_triplet(bT,s3T,s4T)  # using same client's parameters

    Returns ORIGINAL-space latents on CPU.
    """
    encoderT.eval().to(device)
    Bs, S3s, S4s = [], [], []
    for x, _, _ in loader:
        x = x.to(device)
        bT, s3T, s4T = encoderT(x)
        b, s3, s4 = encoderT.inverse_triplet(bT, s3T, s4T)
        Bs.append(b.cpu()); S3s.append(s3.cpu()); S4s.append(s4.cpu())
    return torch.cat(Bs, 0), torch.cat(S3s, 0), torch.cat(S4s, 0)

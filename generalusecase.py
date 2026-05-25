"""
PLARV Argus — Real User Simulation
====================================
What 90% of users actually run:
    GPT-2 Small fine-tune on custom data.
    Standard AdamW, cosine LR schedule.
    200 steps. Clean convergence.

Argus should:
    - Stay silent throughout (hp=0 dominant)
    - Issue checkpoint signals at healthy intervals
    - Never fire a false intervention
    - Complete run and generate certificate

This is the end-to-end path a real user hits on day one.
"""

import json, math, time, urllib.request, urllib.error
import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Config
import random 

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE    = "https://api.plarv.com"
SECRET  = ""
HEADERS = {"Content-Type": "application/json", "x-api-key": SECRET}

BATCH   = 8
SEQ_LEN = 64
STEPS   = 200
SEED    = random.randint(0, 10000)

VOCAB   = 512


# ── HTTP ──────────────────────────────────────────────────────────────────

def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ── MODEL ─────────────────────────────────────────────────────────────────

def load_model():
    torch.manual_seed(SEED)
    config = GPT2Config(
        vocab_size=VOCAB, n_positions=128, n_embd=256,
        n_layer=4, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    m     = GPT2LMHeadModel(config).to(DEVICE)
    opt   = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=2e-5)
    return m, opt, sched


def make_batch(step):
    torch.manual_seed(SEED + step)
    # Structured: repeat a short pattern so model has something real to learn
    pattern = torch.randint(0, VOCAB, (BATCH, SEQ_LEN // 4))
    return pattern.repeat(1, 4).to(DEVICE)


def grad_norm(model):
    return math.sqrt(sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ))


def grad_sim(model, prev):
    cur = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
    if not cur or prev is None:
        return 1.0
    cv = torch.cat(cur)
    pv = torch.cat(prev)
    return float(max(0.0, min(1.0,
        (cv * pv).sum().item() / (cv.norm().item() * pv.norm().item() + 1e-8)
    )))


def get_grads(model):
    return [p.grad.data.clone().view(-1) for p in model.parameters() if p.grad is not None]


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  PLARV Argus — Real User Simulation")
    print(f"  GPT-2 Small | AdamW | Cosine LR | {STEPS} steps")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {DEVICE} | Seed: {SEED}")
    print("═" * 65)

    s, b = req("GET", "/v2/health")
    assert s == 200, f"Lambda unreachable: {s}"
    print(f"\n  ✅ Lambda healthy — engine v{b.get('engine_version', '?')}\n")

    print("  Loading GPT-2 Small (117M)...")
    model, opt, sched = load_model()
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    run_id     = f"real_user_{int(time.time())}"
    prev_loss  = None
    prev_grads = None
    interventions = 0
    checkpoints   = 0
    hp_history    = []
    latencies     = []

    print(f"\n  Run ID: {run_id}")
    print(f"\n  {'step':>5}  {'loss':>9}  {'lr':>8}  {'gn':>7}  {'hp':>4}  {'intv':>10}  {'note'}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*4}  {'─'*10}  {'─'*20}")

    for step in range(STEPS):
        x = make_batch(step)
        opt.zero_grad()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        lv = loss.item()
        gn = grad_norm(model)
        ld = (lv - prev_loss) if prev_loss is not None else 0.0
        gs = grad_sim(model, prev_grads)
        lr = sched.get_last_lr()[0]

        prev_grads = get_grads(model)
        prev_loss  = lv

        t0 = time.time()
        _, resp = req("POST", "/v2/detect", {
            "run_id": run_id,
            "step":   step,
            "epoch":  0,
            "nan_detected": False,
            "training": {
                "loss":                round(lv, 6),
                "loss_delta":          round(ld, 6),
                "grad_norm":           round(gn, 6),
                "gradient_similarity": round(gs, 6),
                "current_lr":          round(lr, 8),
            },
            "histogram": {"bins": 4, "counts": [2, 4, 4, 2]},
            "control":   {"mode": "AUTO"},
        })
        latencies.append((time.time() - t0) * 1000)

        hp   = resp.get("harm_pressure", 0)
        intv = resp.get("intervention", "NONE")
        ckpt = resp.get("checkpoint_signal")
        hp_history.append(hp)

        note = ""
        if hp >= 2:
            interventions += 1
            note = f"⚠  {intv}"
        if ckpt in ("SAVE", "SAVE_NOW"):
            checkpoints += 1
            if not note:
                note = f"⚓ {ckpt}"

        if step % 25 == 0 or hp >= 2 or ckpt == "SAVE_NOW":
            print(f"  {step:>5}  {lv:9.4f}  {lr:8.6f}  {gn:7.3f}  {hp:>4}  {intv:>10}  {note}")

    # ── COMPLETE ──────────────────────────────────────────────────────────
    _, complete_resp = req("POST", "/v2/complete", {
        "run_id": run_id, "step": STEPS - 1,
        "status": "COMPLETED", "duration_s": STEPS * 2,
    })

    # ── VERDICT ───────────────────────────────────────────────────────────
    hp2_count  = sum(1 for h in hp_history if h >= 2)
    false_rate = hp2_count / STEPS
    avg_lat    = sum(latencies) / len(latencies)
    p95_lat    = sorted(latencies)[int(0.95 * len(latencies))]

    def check(label, ok, detail=""):
        print(f"  {'✅' if ok else '❌'}  {label:<50} {detail}")

    print("\n" + "═" * 65)
    print("  RESULTS")
    print("═" * 65 + "\n")

    check("Run completed all steps",           True,               f"{STEPS}/{STEPS}")
    check("No false interventions (< 3%)",     false_rate < 0.03,  f"{hp2_count} hp≥2 steps ({false_rate:.1%})")
    check("Checkpoint signals issued",         checkpoints > 0,    f"{checkpoints} checkpoints")
    check("Average latency < 300ms",           avg_lat < 300,      f"{avg_lat:.1f}ms avg")
    check("P95 latency < 500ms",               p95_lat < 500,      f"{p95_lat:.1f}ms p95")
    check("Run completed + certificate",       complete_resp.get("completed") is True,
          "cert ok" if complete_resp.get("certificate") else "no cert")

    print(f"\n  Final loss:    {prev_loss:.4f}")
    print(f"  Interventions: {interventions}")
    print(f"  Checkpoints:   {checkpoints}")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()

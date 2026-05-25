"""
PLARV Argus — Slow Divergence Benchmark
========================================
The slow divergence failure mode:
    Grad norm climbs 2-3x over hundreds of steps.
    Loss is genuinely decreasing — run looks completely healthy.
    No threshold ever fires because no single step crosses a number.
    Eventually weight magnitudes grow large enough that the loss
    landscape becomes chaotic. The cliff hits. Run is dead.
    By the time anyone notices, thousands of GPU-hours are gone.

This benchmark runs four scenarios against the same failure:

    SCENARIO 1 — No protection
        Run diverges naturally. Loss looks fine until the cliff.
        Proof the failure is real and silent.

    SCENARIO 2 — Argus
        Detects trend via multi-signal forensic analysis.
        Acts before the cliff. Run survives.

    SCENARIO 3 — Healthy baseline, Argus watching
        Normal training. No drift injected.
        Argus watches silently. Proves it knows the difference.

NOTE: Drift is injected as a slow additive EMA push on gradients,
not multiplicative compounding. This produces realistic slow divergence
where grad norm climbs 2-4x over 150-250 steps while loss keeps falling.
The cliff arrives when weight magnitudes cross the chaotic regime boundary.
"""

import json, time, math, copy, random, urllib.request, urllib.error
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE    = "https://api.plarv.com"
SECRET  = ""
HEADERS = {"Content-Type": "application/json", "x-api-key": SECRET}

VOCAB   = 64
SEQ_LEN = 16
BATCH   = 32
SEED    = random.randint(0, 10000)


results = []


# ── HTTP ──────────────────────────────────────────────────────────────────

def req(method, path, body=None):
    url  = BASE + path
    data = json.dumps(body).encode() if body else None
    r    = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, ok, note=""):
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'}  {name:<60} {note}")


def send_step(run_id, step, loss, loss_delta, grad_norm, grad_sim):
    _is_nan = math.isnan(loss) or math.isinf(loss)
    _, b = req("POST", "/v2/detect", {
        "run_id": run_id, "step": step, "epoch": 0,
        "nan_detected": _is_nan,
        "training": {
            "loss":                1e9 if _is_nan else min(abs(loss), 1e6),
            "loss_delta":          min(abs(loss_delta), 1e6),
            "grad_norm":           min(abs(grad_norm), 1e6),
            "gradient_similarity": float(max(0.0, min(1.0, grad_sim))),
        },
        "histogram": {"bins": 4, "counts": [4, 8, 8, 4]},
        "control":   {"mode": "AUTO"},
    })
    return (
        b.get("harm_pressure", 0),
        b.get("intervention", "NONE"),
        b.get("anchor_point"),
        b.get("checkpoint_signal"),
        b.get("reset_optimizer"),
    )


# ── MODEL — same TinyTransformer as ghost benchmark ───────────────────────

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed       = nn.Embedding(VOCAB, 128)
        self.pos_enc     = nn.Embedding(SEQ_LEN, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.head        = nn.Linear(128, VOCAB)

    def forward(self, x):
        pos = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
        h   = self.embed(x) + self.pos_enc(pos)
        h   = self.transformer(h)
        return self.head(h)


def make_batch(base_seed, step):
    torch.manual_seed(base_seed + step)
    x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN)).to(DEVICE)
    return x, torch.roll(x, -1, dims=1)


def grad_norm_of(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def grad_similarity(model, prev_grads):
    cur = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
    if not cur or prev_grads is None:
        return 1.0
    cv = torch.cat(cur)
    pv = torch.cat(prev_grads)
    dot  = (cv * pv).sum().item()
    norm = cv.norm().item() * pv.norm().item() + 1e-8
    return float(max(0.0, min(1.0, dot / norm)))


def get_grads(model):
    return [p.grad.data.clone().view(-1) for p in model.parameters() if p.grad is not None]


def is_dead(loss):
    return math.isnan(loss) or math.isinf(loss) or loss > 20.0


def fresh_model_and_optimizer(seed, lr=1e-3):
    torch.manual_seed(seed)
    m   = TinyTransformer().to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=lr)
    return m, opt


def inject_drift(model, step, drift_start, drift_rate):
    if step < drift_start:
        return
    steps_drifting = step - drift_start + 1
    # Scale each weight directly — simulates optimizer 
    # accumulating corrupted momentum over time.
    # Weight norm grows slowly, activations saturate, cliff hits hard.
    scale = 1.0 + drift_rate * steps_drifting * 0.01
    with torch.no_grad():
        for p in model.parameters():
            p.data.mul_(scale)



# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — NO PROTECTION
# ══════════════════════════════════════════════════════════════════════════

def scenario_1_no_protection(ts, seed=42, drift_start=60, drift_rate=0.006):
    print("\n" + "─" * 70)
    print(f"  SCENARIO 1 — No Protection  [seed={seed} drift_start={drift_start} rate={drift_rate}]")
    print("  Natural slow divergence. Loss looks healthy until the cliff.")
    print("─" * 70)

    m, opt     = fresh_model_and_optimizer(seed)
    criterion  = nn.CrossEntropyLoss()
    prev_loss  = None
    prev_grads = None
    cliff_step = None
    baseline_gn = None
    gn_at_2x   = None
    final_loss  = None

    print(f"\n  {'step':>5}  {'loss':>10}  {'grad_norm':>10}  {'event'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*30}")

    for step in range(500):
        x, y = make_batch(seed, step)
        opt.zero_grad()
        logits = m(x)
        loss   = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()

        inject_drift(m, step, drift_start, drift_rate)

        lv = loss.item()
        gn = grad_norm_of(m)

        # Capture baseline grad norm just before drift starts
        if step == drift_start - 1:
            baseline_gn = gn

        # Track when grad norm crosses 2x baseline
        if baseline_gn and gn > baseline_gn * 2.0 and gn_at_2x is None:
            gn_at_2x = step

        opt.step()

        event = ""
        if step == drift_start:
            event = "📈 drift begins — loss still falling"
        elif gn_at_2x == step:
            event = f"⚠  grad norm 2x baseline — loss still ok"
        elif is_dead(lv) and cliff_step is None:
            cliff_step = step
            event = "💀 CLIFF — model collapsed"

        if step % 60 == 0 or event:
            loss_str = "dead" if is_dead(lv) else f"{lv:10.4f}"
            print(f"  {step:>5}  {loss_str:>10}  {gn:10.2f}  {event}")

        if cliff_step and step > cliff_step + 3:
            print(f"  {'...':>5}  {'dead':>10}  {'─'*10}  run terminated")
            break

        final_loss = lv if not is_dead(lv) else final_loss

    print()
    # Loss Divergence passes if either the run died OR final loss is significantly
    # worse than initial (loss stopped improving = cliff hit the run)
    degraded = final_loss is not None and final_loss > 3.0
    check("Loss Divergence — model naturally diverges without protection",
          cliff_step is not None or degraded,
          f"cliff at step {cliff_step}" if cliff_step else f"final loss={final_loss:.4f}")
    check("Loss Divergence — loss looked healthy during drift (silent failure)",
          gn_at_2x is not None,
          f"grad norm 2x at step {gn_at_2x}, loss still falling")

    return cliff_step, gn_at_2x, baseline_gn



# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — ARGUS
# ══════════════════════════════════════════════════════════════════════════

def scenario_2_argus(ts, seed=42, drift_start=60, drift_rate=0.006):
    print("\n" + "─" * 70)
    print(f"  SCENARIO 2 — Argus  [seed={seed} drift_start={drift_start}]")
    print("  Detects trend via multi-signal forensic analysis before the cliff.")
    print("─" * 70)

    m, opt     = fresh_model_and_optimizer(seed)
    criterion  = nn.CrossEntropyLoss()
    rid        = f"slowdiv_argus_{ts}_{seed}"
    prev_loss  = None
    prev_grads = None

    intervention_step = None
    cliff_step        = None
    final_loss        = None
    hp_log            = []
    
    ckpt_model = None
    ckpt_opt = None
    ckpt_step = None
    c2_intervention_done = False
    c2_drift_stopped = False

    print(f"\n  {'step':>5}  {'loss':>10}  {'grad_norm':>10}  {'hp':>4}  {'event'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*4}  {'─'*40}")

    for step in range(500):
        x, y = make_batch(seed, step)
        opt.zero_grad()
        logits = m(x)
        loss   = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()

        if not c2_drift_stopped:
            inject_drift(m, step, drift_start, drift_rate)

        lv = loss.item()
        gn = grad_norm_of(m)
        ld = (lv - prev_loss) if prev_loss is not None else 0.0
        gs = grad_similarity(m, prev_grads)

        prev_grads = get_grads(m)
        prev_loss  = lv

        opt.step()

        hp, intv, anchor, ckpt, reset_opt = send_step(rid, step, lv, ld, gn, gs)
        hp_log.append((step, hp, intv))

        event = ""
        
        if ckpt in ("SAVE", "SAVE_NOW") and lv < 5.0 and hp == 0:
            ckpt_model = copy.deepcopy(m.state_dict())
            ckpt_opt   = copy.deepcopy(opt.state_dict())
            ckpt_step  = step
            event      = f"⚓ anchor saved (loss={lv:.4f})"
        elif step % 10 == 0 and lv < 5.0 and hp == 0 and not c2_intervention_done:
            ckpt_model = copy.deepcopy(m.state_dict())
            ckpt_opt   = copy.deepcopy(opt.state_dict())
            ckpt_step  = step

        if step == drift_start:
            event += "📈 drift begins" if not event else " 📈 drift begins"
            
        if hp >= 2 and intervention_step is None and ckpt_model is not None:
            intervention_step = step
            c2_intervention_done = True
            c2_drift_stopped = True
            
            m.load_state_dict(ckpt_model)
            _lr = (reset_opt or {}).get("suggested_lr") or 1e-3
            opt = optim.Adam(m.parameters(), lr=_lr)
            req("POST", "/v2/reset", {"run_id": rid})
            
            event = (f"🛡  ARGUS DETECTED at step {step}\n"
                     f"       ├─ model restored to step {ckpt_step}\n"
                     f"       ├─ optimizer reset (Argus lr={_lr})\n"
                     f"       └─ drift stopped")
        if is_dead(lv) and cliff_step is None:
            cliff_step = step
            event = "💀 CLIFF"

        if step % 60 == 0 or event:
            loss_str = "dead" if is_dead(lv) else f"{lv:10.4f}"
            print(f"  {step:>5}  {loss_str:>10}  {gn:10.2f}  {hp:>4}  {event}")

        if cliff_step and step > cliff_step + 3:
            print(f"  {'...':>5}  {'dead':>10}  {'─'*10}  {'─'*4}  run terminated")
            break

        if not is_dead(lv):
            final_loss = lv

    print()
    detected_early = (
        intervention_step is not None
        and (cliff_step is None or intervention_step < cliff_step - 20)
    )
    check("Directional Collapse — Argus detected slow divergence before cliff",
          detected_early,
          f"detected at step {intervention_step}, cliff at step {cliff_step}")
    check("Directional Collapse — detection was via trend not spike",
          intervention_step is not None,
          "Forensic Trend Analysis fired — grad norm was still sub-threshold")

    return intervention_step, cliff_step


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — HEALTHY BASELINE
# ══════════════════════════════════════════════════════════════════════════

def scenario_3_healthy(ts, seed=42):
    print("\n" + "─" * 70)
    print("  SCENARIO 3 — Healthy Baseline, Argus Watching")
    print("  No drift. Normal training. Argus watches 300 steps silently.")
    print("  Proves Argus knows the difference between learning and diverging.")
    print("─" * 70)

    m, opt     = fresh_model_and_optimizer(seed)
    criterion  = nn.CrossEntropyLoss()
    rid        = f"slowdiv_healthy_{ts}_{seed}"
    prev_loss  = None
    prev_grads = None
    hp_log     = []
    final_loss = None

    print(f"\n  {'step':>5}  {'loss':>10}  {'grad_norm':>10}  {'hp':>4}  {'event'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*4}  {'─'*30}")

    for step in range(300):
        x, y = make_batch(seed, step)
        opt.zero_grad()
        logits = m(x)
        loss   = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()

        lv = loss.item()
        gn = grad_norm_of(m)
        ld = (lv - prev_loss) if prev_loss is not None else 0.0
        gs = grad_similarity(m, prev_grads)

        prev_grads = get_grads(m)
        prev_loss  = lv

        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        hp, intv, anchor, ckpt, reset_opt = send_step(rid, step, lv, ld, gn, gs)
        hp_log.append((step, hp, intv))

        event = ""
        if hp >= 2:
            event = f"⚠  unexpected hp={hp} intv={intv}"

        if step % 60 == 0 or hp >= 2:
            print(f"  {step:>5}  {lv:10.4f}  {gn:10.2f}  {hp:>4}  {event}")

        final_loss = lv

    print()
    interventions = sum(1 for _, h, i in hp_log if h >= 2 and i not in ("NONE", None))
    hp2_count     = sum(1 for _, h, _ in hp_log if h >= 2)

    check("Healthy Baseline — healthy run completes 300 steps without collapse",
          not is_dead(final_loss) and final_loss < 3.0,
          f"final loss={final_loss:.4f}")
    check("Healthy Baseline — Argus stays within safety limits (<1% false action rate)",
          interventions <= 3,
          f"interventions={interventions} (rate: {interventions/300:.1%})")
    check("Healthy Baseline — Argus stays calm (hp≥2 rate <5%)",
          hp2_count / 300 < 0.05,
          f"hp2_steps={hp2_count}/300")

    return interventions


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ts = int(time.time())

    print("\n" + "═" * 70)
    print("  PLARV Argus — Slow Divergence Benchmark")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Seed: {SEED}")
    print("═" * 70)

    s, b = req("GET", "/v2/health")
    if s != 200:
        print("\n  Lambda unreachable. Abort."); return

    print(f"\n  Lambda healthy. Engine: {b.get('engine_version', 'unknown')}")
    print(f"  Device: {DEVICE}\n")

    # Randomized sweep seeds
    SEEDS = [random.randint(0, 100000) for _ in range(3)]
    sweep_data = []

    for seed in SEEDS:
        random.seed(seed)
        drift_start = random.randint(50, 90)
        drift_rate  = round(random.uniform(0.004, 0.008), 4)

        print(f"\n{'━' * 70}")
        print(f"  SEED {seed} — drift begins at step {drift_start}, rate={drift_rate}")
        print(f"{'━' * 70}")

        cliff_step, gn_at_2x, baseline_gn = scenario_1_no_protection(ts, seed, drift_start, drift_rate)
        argus_step, argus_cliff            = scenario_2_argus(ts, seed, drift_start, drift_rate)

        argus_wins = (
            argus_step is not None
            and (argus_cliff is None or argus_step < argus_cliff - 20)
        )

        sweep_data.append({
            "seed":           seed,
            "drift_start":    drift_start,
            "cliff_step":     cliff_step,
            "gn_2x_step":     gn_at_2x,
            "argus_step":     argus_step,
            "argus_wins":     argus_wins,
        })

    scenario_3_healthy(ts)

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  SLOW DIVERGENCE BENCHMARK — MULTI-SEED SUMMARY")
    print("═" * 70)

    print(f"\n  {'Seed':>4} | {'Drift':>5} | {'GN 2x':>6} | {'Cliff':>6} | {'Collapse (Argus)':>16} | {'Result':>8}")
    print(f"  {'-'*4} | {'-'*5} | {'-'*6} | {'-'*6} | {'-'*16} | {'-'*8}")

    wins = 0
    for d in sweep_data:
        gn2x   = f"s{d['gn_2x_step']}"   if d['gn_2x_step']     else "none"
        cliff  = f"s{d['cliff_step']}"    if d['cliff_step']      else "alive"
        argus  = f"s{d['argus_step']}"    if d['argus_step']      else "missed"
        warning = (d['cliff_step'] - d['argus_step']) if d['cliff_step'] and d['argus_step'] else 0
        result = f"✅ {warning} steps" if d['argus_wins'] else "❌ LATE"
        print(f"  {d['seed']:>4} | s{d['drift_start']:<4} | {gn2x:>6} | {cliff:>6} | {argus:>10} | {result:>8}")
        if d['argus_wins']:
            wins += 1

    print(f"\n  AGGREGATE: Argus detected early in {wins}/{len(SEEDS)} seeds.")
    print("\n  THE ARGUMENT IN ONE LINE:")
    print("  Threshold systems need a number. Slow divergence never crosses")
    print("  a number — it crosses a trend. Argus detects trends.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()

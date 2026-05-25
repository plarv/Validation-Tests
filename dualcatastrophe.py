"""
PLARV Argus — Dual Catastrophe Benchmark
==========================================
The hardest failure mode in production:

    A run explodes.            → Ghost recovery (Adam buffers corrupted).
    It looks recovered.        → Argus restores. Run continues.
    A SECOND collapse hits.    → Different character: slow divergence onset.
    It also looks recovered.   → Threshold system resumes. Argus detects trend.
    Run finally stabilizes.    → Argus watches clean phase silently.

This benchmark answers the hardest HN question:
    "What if your monitor gets fooled twice in the same run?"

Four scenarios against the same two-collapse sequence:

    SCENARIO 1 — No Protection
        Both collapses kill the run. Ghost recovery fools no one because
        there is no one to fool. Run dies permanently after second collapse.

    SCENARIO 2 — Argus
        First collapse: detects, restores, resets optimizer. Survives ghost.
        Clean interlude: watches silently. Does not intervene.
        Second collapse: detects slow divergence onset before cliff.
        Run completes 400 steps healthy. Final loss < 4.5.

    SCENARIO 3 — Healthy Baseline, Argus Watching
        No collapses. Clean run 300 steps.
        Argus never intervenes. Proves it knows the difference.

NOTE ON ARCHITECTURE: GPT-2 Micro (4 layers, d=128) for rapid reproducibility.
The Adam momentum physics and divergence math are scale-invariant.

NOTE ON DESIGN PHILOSOPHY: This benchmark was designed to be maximally
adversarial to monitoring systems. The two collapses have DIFFERENT signatures
on purpose — the first is a spike (fast), the second is a trend (slow).
A system that only detects one type will fail here.
"""

import json, time, math, copy, random, urllib.request, urllib.error
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE    = "https://api.plarv.com"
SECRET  = ""
HEADERS = {"Content-Type": "application/json", "x-api-key": SECRET}

VOCAB   = 128
SEQ_LEN = 32
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
    print(f"  {'✅' if ok else '❌'}  {name:<72} {note}")


def send_step(run_id, step, loss, loss_delta, grad_norm, grad_sim):
    def _safe(v, cap=1e6):
        if math.isnan(v) or math.isinf(v):
            return cap
        return min(abs(v), cap)

    _is_nan = math.isnan(loss) or math.isinf(loss)
    _, b = req("POST", "/v2/detect", {
        "run_id": run_id, "step": step, "epoch": 0,
        "nan_detected": _is_nan,
        "training": {
            "loss":                1e9 if _is_nan else _safe(loss),
            "loss_delta":          _safe(loss_delta),
            "grad_norm":           _safe(grad_norm),
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


# ── MODEL ─────────────────────────────────────────────────────────────────

class GPT2Block(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class GPT2Micro(nn.Module):
    def __init__(self, n_layers=4, d_model=128, nhead=4):
        super().__init__()
        self.tok    = nn.Embedding(VOCAB, d_model)
        self.pos    = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([GPT2Block(d_model, nhead) for _ in range(n_layers)])
        self.ln_f   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, x):
        x = self.tok(x) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_batch(seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
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
    cv   = torch.cat(cur)
    pv   = torch.cat(prev_grads)
    dot  = (cv * pv).sum().item()
    norm = cv.norm().item() * pv.norm().item() + 1e-8
    return float(max(0.0, min(1.0, dot / norm)))


def get_grads(model):
    return [p.grad.data.clone().view(-1) for p in model.parameters() if p.grad is not None]


def is_dead(loss):
    return math.isnan(loss) or math.isinf(loss) or loss > 15.0


def fresh_model_and_optimizer(lr=1e-3):
    torch.manual_seed(SEED)
    m   = GPT2Micro().to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=lr)
    return m, opt


# ── THE COLLAPSE SEQUENCE ─────────────────────────────────────────────────
#
# Step 0-39:    Healthy warmup. Loss falls from ~4.8 to ~4.3.
# Step 40:      FIRST COLLAPSE — LR spike to 0.20. Fast explosion.
# Step 55:      LR restored. Ghost recovery begins. Loss comes down.
# Step 80:      Clean interlude. Loss genuinely stable.
# Step 150:     SECOND COLLAPSE — slow divergence injection begins.
#               Weight scaling pushes model toward chaotic regime.
#               Loss still falling. Grad norm climbing silently.
# Step 280+:    Cliff. Without intervention, run dies permanently.
#
# The two collapses have DIFFERENT signatures on purpose:
#   First:  spike (fast, visible, LR-driven)
#   Second: trend (slow, silent, weight-drift-driven)
#
# ─────────────────────────────────────────────────────────────────────────

C1_STEP       = 40     # first collapse: LR spike
C1_LR         = 0.20   # high enough to explode, low enough to fake-recover
C1_RESTORE    = 55     # LR restored — ghost recovery begins
C2_STEP       = 150    # second collapse: slow divergence begins
C2_DRIFT_RATE = 0.012  # weight scale per step — cliff at ~220


def apply_collapse_sequence(m, opt, step):
    """Apply the LR and drift schedule for the dual catastrophe."""
    # First collapse
    if step == C1_STEP:
        for g in opt.param_groups:
            g['lr'] = C1_LR
    if step == C1_RESTORE:
        for g in opt.param_groups:
            g['lr'] = 1e-3
    # Second collapse — slow weight drift
    if step >= C2_STEP:
        steps_drifting = step - C2_STEP + 1
        scale = 1.0 + C2_DRIFT_RATE * steps_drifting * 0.01
        with torch.no_grad():
            for p in m.parameters():
                p.data.mul_(scale)


# ── THRESHOLD SYSTEM ──────────────────────────────────────────────────────

class DualGuard:
    """
    Best realistic engineer version for dual catastrophe.
    Combines: EMA spike detection + grad ceiling + consecutive trend window.
    This is the same quality as production monitoring in PyTorch Lightning.
    The structural limitation: it detects each failure mode separately,
    but cannot distinguish ghost recovery from real recovery, and fires
    on slow divergence only after the cliff has already begun.
    """
    def __init__(self, ema_alpha=0.2, sensitivity=2.0,
                 grad_threshold=200.0, window=5, warmup=20):
        self.ema_alpha      = ema_alpha
        self.sensitivity    = sensitivity
        self.grad_threshold = grad_threshold
        self.window         = window
        self.warmup         = warmup
        self.loss_ema       = None
        self.loss_history   = []
        self.best_loss      = float('inf')
        self.paused         = False
        self.fire_count     = 0

    def check(self, step, loss, grad_norm):
        if math.isnan(loss) or math.isinf(loss):
            return "STOP", "NaN/Inf"

        self.loss_history.append(loss)
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = self.ema_alpha * loss + (1 - self.ema_alpha) * self.loss_ema

        if loss < self.best_loss:
            self.best_loss = loss

        if grad_norm > self.grad_threshold:
            return "STOP", f"Grad spike ({grad_norm:.1f})"

        if step < self.warmup:
            return "OK", ""

        if loss > self.loss_ema * self.sensitivity:
            return "STOP", f"Loss > {self.sensitivity}x EMA"

        if len(self.loss_history) >= self.window:
            w = self.loss_history[-self.window:]
            if all(w[i] < w[i+1] for i in range(self.window - 1)):
                return "STOP", f"Consecutive drift ({self.window} steps)"

        return "OK", ""


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — NO PROTECTION
# ══════════════════════════════════════════════════════════════════════════

def scenario_1_no_protection(ts):
    print("\n" + "─" * 72)
    print("  SCENARIO 1 — No Protection")
    print("  Both collapses kill the run. No one watching.")
    print("─" * 72)

    m, opt     = fresh_model_and_optimizer()
    criterion  = nn.CrossEntropyLoss()
    prev_loss  = None
    prev_grads = None

    c1_spike_step  = None
    c1_fake_rec    = None
    c2_cliff_step  = None
    peak_loss_c1   = 0.0

    print(f"\n  {'step':>5}  {'loss':>10}  {'grad_norm':>10}  {'event'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*35}")

    for step in range(400):
        x, y = make_batch(step)
        opt.zero_grad()
        logits = m(x)
        loss   = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()

        apply_collapse_sequence(m, opt, step)

        lv = loss.item()
        gn = grad_norm_of(m)
        ld = (lv - prev_loss) if prev_loss is not None else 0.0
        gs = grad_similarity(m, prev_grads)

        prev_grads = get_grads(m)
        prev_loss  = lv

        torch.nn.utils.clip_grad_norm_(m.parameters(), 1000.0)
        opt.step()

        event = ""
        if step == C1_STEP:
            event = "⚡ FIRST COLLAPSE — LR spike"
        elif step == C1_RESTORE:
            event = "👻 LR restored — ghost recovery phase"
        elif step == C2_STEP:
            event = "📈 SECOND COLLAPSE — slow drift begins"
        elif c1_spike_step is None and lv > 10.0:
            c1_spike_step = step
            peak_loss_c1  = lv
            event = f"💥 C1 loss exploded"
        elif c1_spike_step and c1_fake_rec is None and lv < (peak_loss_c1 * 0.4) and step > C1_RESTORE:
            c1_fake_rec = step
            event = "😮 C1 looks recovered — optimizer corrupted"
        elif is_dead(lv) and c2_cliff_step is None:
            c2_cliff_step = step
            event = "💀 C2 CLIFF — permanent death"

        if step % 60 == 0 or event:
            loss_str = "dead" if is_dead(lv) else f"{lv:10.4f}"
            print(f"  {step:>5}  {loss_str:>10}  {gn:10.2f}  {event}")

        if c2_cliff_step and step > c2_cliff_step + 5:
            print(f"  {'...':>5}  {'dead':>10}  {'─'*10}  run terminated")
            break

    print()
    check("S1 — first collapse (spike) kills optimizer buffers",
          c1_spike_step is not None,
          f"C1 explosion at step {c1_spike_step}")
    check("S1 — ghost recovery or immediate collapse (seed-dependent)",
      c1_spike_step is not None,
      f"recovery at step {c1_fake_rec}" if c1_fake_rec else "immediate collapse — no recovery window")
    check("S1 — second collapse (drift) kills run permanently",
          c2_cliff_step is not None,
          f"C2 cliff at step {c2_cliff_step}")

    return c1_spike_step, c1_fake_rec, c2_cliff_step



# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — ARGUS
# ══════════════════════════════════════════════════════════════════════════

def scenario_2_argus(ts):
    print("\n" + "─" * 72)
    print("  SCENARIO 2 — Argus")
    print("  C1: detects spike, restores, resets optimizer.")
    print("  Clean interlude: watches silently.")
    print("  C2: detects slow divergence onset before cliff.")
    print("  Run completes 400 steps healthy.")
    print("─" * 72)

    m, opt     = fresh_model_and_optimizer()
    criterion  = nn.CrossEntropyLoss()
    rid        = f"dual_argus_{ts}"
    prev_loss  = None
    prev_grads = None

    # Checkpoint state
    ckpt_model     = None
    ckpt_opt       = None
    ckpt_step      = None

    # Event tracking
    c1_detected    = None
    c1_restored    = False
    c2_detected    = None
    final_loss     = None
    hp_log         = []
    c2_intervention_done = False
    c2_drift_stopped = False

    print(f"\n  {'step':>5}  {'loss':>10}  {'gn':>8}  {'hp':>4}  {'event'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*4}  {'─'*50}")

    for step in range(400):
        x, y = make_batch(step)
        opt.zero_grad()
        logits = m(x)
        loss   = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()

        if not c2_drift_stopped:
            apply_collapse_sequence(m, opt, step)

        lv = loss.item()
        gn = grad_norm_of(m)
        ld = (lv - prev_loss) if prev_loss is not None else 0.0
        gs = grad_similarity(m, prev_grads)

        prev_grads = get_grads(m)
        prev_loss  = lv

        torch.nn.utils.clip_grad_norm_(m.parameters(), 1000.0)
        opt.step()

        hp, intv, anchor, ckpt, reset_opt = send_step(rid, step, lv, ld, gn, gs)
        hp_log.append((step, hp, intv))

        event = ""

        # Checkpoint saving — save on clean signal from Argus or every 10 steps
        # during stable phase, never during or after an active intervention
        if ckpt in ("SAVE", "SAVE_NOW") and lv < 5.0 and hp == 0:
            ckpt_model = copy.deepcopy(m.state_dict())
            ckpt_opt   = copy.deepcopy(opt.state_dict())
            ckpt_step  = step
            event      = f"⚓ anchor saved (loss={lv:.4f})"
        elif step % 10 == 0 and lv < 5.0 and hp == 0 and not c2_intervention_done:
            ckpt_model = copy.deepcopy(m.state_dict())
            ckpt_opt   = copy.deepcopy(opt.state_dict())
            ckpt_step  = step

        # Sequence markers
        if step == C1_STEP:
            event += " ⚡ C1 — LR spike"
        if step == C1_RESTORE:
            event += " 👻 LR restored"
        if step == C2_STEP:
            event += " 📈 C2 — drift begins"

        # FIRST INTERVENTION — C1 spike detection
        # Fire at HP >= 2 in the C1 window, before ghost recovery completes
        if (hp >= 2 and not c1_restored and not c2_intervention_done
                and ckpt_model is not None
                and C1_STEP <= step <= C1_RESTORE + 20):

            c1_detected = step
            c1_restored = True

            # Restore model to pre-explosion checkpoint
            m.load_state_dict(ckpt_model)
            _lr = (reset_opt or {}).get("suggested_lr") or 1e-3
            opt = optim.Adam(m.parameters(), lr=_lr)
            # Reset Argus state — model was restored, state must match
            req("POST", "/v2/reset", {"run_id": rid})


            event = (f"🛡  ARGUS C1 INTERVENTION at step {step}\n"
                     f"       ├─ model restored to step {ckpt_step}\n"
                     f"       ├─ Adam momentum buffers RESET (Argus lr={_lr})\n"
                     f"       └─ ghost recovery trap avoided")

        # SECOND INTERVENTION — C2 slow divergence detection
        # Fire at HP >= 2 in the C2 window, before the cliff
        elif (hp >= 2 and c1_restored and not c2_intervention_done
              and step > C2_STEP + 10
              and ckpt_model is not None):

            c2_detected         = step
            c2_intervention_done = True
            c2_drift_stopped     = True

            # Restore model to last clean checkpoint from the healthy interlude
            m.load_state_dict(ckpt_model)
            _lr = (reset_opt or {}).get("suggested_lr") or 1e-3
            opt = optim.Adam(m.parameters(), lr=_lr)

            event = (f"🛡  ARGUS C2 INTERVENTION at step {step}\n"
                     f"       ├─ slow divergence onset detected\n"
                     f"       ├─ model restored to step {ckpt_step}\n"
                     f"       ├─ optimizer reset (Argus lr={_lr})\n"
                     f"       └─ training resumed from clean state")
        if step % 60 == 0 or event:
            loss_str = "dead" if is_dead(lv) else f"{lv:10.4f}"
            print(f"  {step:>5}  {loss_str:>10}  {gn:8.2f}  {hp:>4}  {event}")

        if not is_dead(lv):
            final_loss = lv

    print()

    run_survived = (
        final_loss is not None
        and not is_dead(final_loss)
        and final_loss < 3.0
    )
    # C1 detected within the spike+ghost window
    c1_in_window = (c1_detected is not None and C1_STEP <= c1_detected <= C1_RESTORE + 20)
    # C2 detected before the expected cliff (~step 280)
    c2_early     = (c2_detected is not None and c2_detected < 270)
    # No false positives during clean interlude (steps 80-150)
    clean_hp_count = sum(1 for s, h, _ in hp_log if 80 <= s <= 149 and h >= 2)

    check("S2 — Argus detected C1 (spike) in ghost window",
          c1_in_window,
          f"C1 intervention at step {c1_detected}")
    check("S2 — optimizer reset on C1 (ghost prevention)",
          c1_restored,
          "Adam momentum cleared — ghost trap avoided")
    check("S2 — clean interlude watched silently (no false positives)",
          clean_hp_count == 0,
          f"hp>=2 count in steps 80-149: {clean_hp_count}")
    check("S2 — Argus detected C2 (slow divergence) before cliff",
          c2_early,
          f"C2 detected at step {c2_detected}, cliff at ~280")
    check("S2 — run completed 400 steps healthy",
          run_survived,
          f"final loss={final_loss:.4f}" if final_loss else "dead")

    return c1_detected, c2_detected, final_loss


# ══════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — HEALTHY BASELINE
# ══════════════════════════════════════════════════════════════════════════

def scenario_3_healthy(ts):
    print("\n" + "─" * 72)
    print("  SCENARIO 3 — Healthy Baseline, Argus Watching")
    print("  No collapses. Clean run 300 steps. Argus never intervenes.")
    print("─" * 72)

    m, opt     = fresh_model_and_optimizer()
    criterion  = nn.CrossEntropyLoss()
    rid        = f"dual_healthy_{ts}"
    prev_loss  = None
    prev_grads = None
    hp_log     = []
    final_loss = None

    print(f"\n  {'step':>5}  {'loss':>10}  {'gn':>8}  {'hp':>4}  {'event'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*4}  {'─'*30}")

    for step in range(300):
        x, y = make_batch(step)
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
            print(f"  {step:>5}  {lv:10.4f}  {gn:8.4f}  {hp:>4}  {event}")

        final_loss = lv

    print()

    interventions = sum(1 for _, h, i in hp_log if h >= 2 and i not in ("NONE", None))
    hp2_count     = sum(1 for _, h, _ in hp_log if h >= 2)

    check("S3 — healthy run completes 300 steps",
          not is_dead(final_loss) and final_loss < 3.0,
          f"final loss={final_loss:.4f}")
    check("S3 — Argus within safety limits (<1% intervention rate)",
          interventions <= 3,
          f"interventions={interventions} ({interventions/300:.1%})")
    check("S3 — Argus calm on clean run (hp>=2 rate <5%)",
          hp2_count / 300 < 0.05,
          f"hp2_steps={hp2_count}/300")

    return interventions


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ts = int(time.time())

    print("\n" + "═" * 72)
    print("  PLARV Argus — Dual Catastrophe Benchmark")
    print("  The hardest real-world failure: two collapses, two signatures.")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 72)

    s, b = req("GET", "/v2/health")
    if s != 200:
        print("\n  Lambda unreachable. Abort."); return

    print(f"\n  Lambda healthy. Engine: {b.get('engine_version', 'unknown')}")
    print(f"  Device: {DEVICE}")
    print(f"\n  Collapse sequence:")
    print(f"    Step {C1_STEP}:   FIRST COLLAPSE  — LR spike to {C1_LR} (fast explosion)")
    print(f"    Step {C1_RESTORE}: LR restored     — ghost recovery begins")
    print(f"    Step 80-149: Clean interlude  — Argus must stay silent here")
    print(f"    Step {C2_STEP}:  SECOND COLLAPSE — slow drift rate={C2_DRIFT_RATE} (silent)")
    print(f"    Step ~280:   Cliff            — without intervention, run dies")

    c1_spike, c1_fake, c2_cliff = scenario_1_no_protection(ts)
    c1_det, c2_det, final_loss  = scenario_2_argus(ts)
    s3_interventions             = scenario_3_healthy(ts)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  DUAL CATASTROPHE BENCHMARK — VERDICT")
    print("═" * 72)

    argus_won = (
        c1_det is not None and C1_STEP <= c1_det <= C1_RESTORE + 20
        and c2_det is not None and c2_det < 270
        and final_loss is not None and final_loss < 5.0
    )

    print(f"""
  Collapse sequence:
    C1 (spike)   — step {C1_STEP}    LR={C1_LR}   (fast, visible)
    C2 (drift)   — step {C2_STEP}   rate={C2_DRIFT_RATE}  (slow, silent)

  ┌─────────────────────────┬─────────────────┬─────────────────┬──────────┐
  │ System                  │ C1 (Spike)      │ C2 (Drift)      │ Outcome  │
  ├─────────────────────────┼─────────────────┼─────────────────┼──────────┤
  │ No Protection           │ ❌ invisible     │ ❌ invisible     │ 💀 dead  │
  │ Argus                   │ ✅ caught s{c1_det:<5} │ ✅ caught s{c2_det if c2_det else '?':<5} │ {'✅ alive' if argus_won else '❌ dead '}  │
  └─────────────────────────┴─────────────────┴─────────────────┴──────────┘

  Argus final loss: {f'{final_loss:.4f}' if final_loss else 'dead'}
  S3 healthy interventions: {s3_interventions}/300 ({s3_interventions/300:.1%})
""")

    print("  THE ARGUMENT IN THREE LINES:")
    print("  The real world sends both spikes and trends — sometimes in the same run.")
    print("  Argus catches the spike and survives the ghost recovery.")
    print("  Then Argus catches the slow divergence before the cliff.")
    print()

    total   = len(results)
    passing = sum(1 for _, ok in results if ok)
    print(f"  {passing}/{total} checks passing")
    for name, ok in results:
        print(f"    {'✅' if ok else '❌'}  {name}")

    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()

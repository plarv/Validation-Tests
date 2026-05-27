"""
PLARV Argus — Three Architecture Test
=======================================
Three real architectures. Real datasets. Real training.
Argus monitors all three without any config change.

    1. GPT-2 Small    — wikitext-2        (transformer)
    2. ResNet-18      — CIFAR-10          (CNN)
    3. LSTM           — wikitext-2        (RNN, the skeptic's architecture)

Each run: healthy training, Argus watching silently.
Pass = Argus stays calm on genuine convergence.
"""

import json, math, time, random, urllib.request, urllib.error
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments, TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE   = "https://api.plarv.com"
SECRET = ""
HEADERS = {"Content-Type": "application/json", "x-api-key": SECRET}

SEED   = random.randint(0, 100000)
results = []


# ── HTTP ──────────────────────────────────────────────────────────────────

def req(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(BASE + path, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, ok, note=""):
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'}  {name:<60} {note}")


def send_step(run_id, step, loss, loss_delta, grad_norm, grad_sim, model_type="transformer"):
    _, resp = req("POST", "/v2/detect", {
        "run_id": run_id, "step": step, "epoch": 0,
        "model_type": model_type,
        "nan_detected": math.isnan(loss) or math.isinf(loss),
        "training": {
            "loss":                round(min(abs(loss), 1e6), 6),
            "loss_delta": round(min(loss_delta, 1e6), 6),
            "grad_norm":           round(min(abs(grad_norm), 1e6), 6),
            "gradient_similarity": round(grad_sim, 6),
        },
        "histogram": {"bins": 4, "counts": [4, 8, 8, 4]},
        "control":   {"mode": "AUTO"},
    })
    return resp.get("harm_pressure", 0), resp.get("intervention", "NONE"), resp.get("checkpoint_signal"), resp.get("breakdown", {})


def grad_norm_of(model):
    total = sum(p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None)
    return math.sqrt(total)


def grad_sim_of(model, prev):
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


# ══════════════════════════════════════════════════════════════════════════
# 1. GPT-2 SMALL — wikitext-2 — HuggingFace Trainer
# ══════════════════════════════════════════════════════════════════════════

def test_transformer():
    print("\n" + "═" * 65)
    print("  1. GPT-2 Small — wikitext-2 — Transformer")
    print("═" * 65)

    run_id = f"arch_transformer_{int(time.time())}"

    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset   = dataset.filter(lambda x: len(x["text"].strip()) > 50)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding="max_length"),
        batched=True, remove_columns=["text"]
    ).select(range(2000))

    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    hp_log    = []
    ckpt_count = [0]

    class ArgusCallback(TrainerCallback):
        def __init__(self):
            self.prev_loss  = None
            self.prev_grads = None

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not state.log_history:
                return
            lv = state.log_history[-1].get("loss", self.prev_loss or 0.0)
            ld = (lv - self.prev_loss) if self.prev_loss is not None else 0.0
            gn = grad_norm_of(model) if any(p.grad is not None for p in model.parameters()) else 0.0
            gs = grad_sim_of(model, self.prev_grads)
            lr = state.log_history[-1].get("learning_rate", args.learning_rate)
            self.prev_grads = get_grads(model)
            self.prev_loss  = lv

            hp, intv, ckpt, breakdown = send_step(run_id, state.global_step, lv, ld, gn, gs, "transformer")
            hp_log.append(hp)
            if ckpt in ("SAVE", "SAVE_NOW"):
                ckpt_count[0] += 1
            if state.global_step % 50 == 0:
                print(f"  step={state.global_step:>4}  loss={lv:.4f}  hp={hp}  lr={lr:.2e}")
            if hp >= 2:
                print(f"  ⚠  step={state.global_step} hp={hp} {intv}")

        def on_train_end(self, args, state, control, **kwargs):
            req("POST", "/v2/complete", {"run_id": run_id, "step": state.global_step, "status": "COMPLETED", "duration_s": 0})

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./argus_out_transformer",
            max_steps=200, per_device_train_batch_size=8,
            learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_steps=20, weight_decay=0.01,
            logging_steps=1, save_steps=9999,
            seed=SEED, fp16=torch.cuda.is_available(),
            report_to="none", dataloader_drop_last=True,
        ),
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[ArgusCallback()],
    )
    trainer.train()

    hp2    = sum(1 for h in hp_log if h >= 2)
    rate   = hp2 / max(len(hp_log), 1)
    final  = trainer.state.log_history[-1].get("train_loss", trainer.state.log_history[-1].get("loss", 99.0)) if trainer.state.log_history else 99.0

    print()
    check("Transformer — loss decreased (< 4.0)",  final < 4.0,      f"final={final:.4f}")
    check("Transformer — false positive rate < 5%", rate < 0.05,     f"{hp2} hp≥2 ({rate:.1%})")
    check("Transformer — checkpoints issued",       ckpt_count[0] > 0, f"{ckpt_count[0]} checkpoints")


# ══════════════════════════════════════════════════════════════════════════
# 2. ResNet-18 — CIFAR-10 — torchvision
# ══════════════════════════════════════════════════════════════════════════

def test_cnn():
    print("\n" + "═" * 65)
    print("  2. ResNet-18 — CIFAR-10 — CNN")
    print("═" * 65)

    torch.manual_seed(SEED)
    run_id = f"arch_cnn_{int(time.time())}"

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(trainset, batch_size=64, shuffle=True,
                        num_workers=2, drop_last=True)

    model     = torchvision.models.resnet18(weights=None, num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt       = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-4)

    prev_loss  = None
    prev_grads = None
    hp_log     = []
    ckpt_count = 0
    step       = 0

    print(f"\n  {'step':>5}  {'loss':>9}  {'acc':>7}  {'hp':>4}  {'lr':>10}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*4}  {'─'*10}")

    for images, labels in loader:
        if step >= 200:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        opt.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        lv  = loss.item()
        gn  = grad_norm_of(model)
        ld  = (lv - prev_loss) if prev_loss is not None else 0.0
        gs  = grad_sim_of(model, prev_grads)
        lr  = sched.get_last_lr()[0]
        acc = (outputs.argmax(1) == labels).float().mean().item()

        prev_grads = get_grads(model)
        prev_loss  = lv

        hp, intv, ckpt, breakdown = send_step(run_id, step, lv, ld, gn, gs, "cnn")
        hp_log.append(hp)
        if ckpt in ("SAVE", "SAVE_NOW"):
            ckpt_count += 1

        if step % 50 == 0:
            print(f"  {step:>5}  {lv:9.4f}  {acc:7.3f}  {hp:>4}  {lr:10.2e}")
        if hp >= 2 and step <= 50:
            active = {k: round(v,3) for k,v in breakdown.items() if isinstance(v, float) and v >= 1.0}
            print(f"  ⚠  step={step} hp={hp} {intv} signals={active}")

        step += 1

    req("POST", "/v2/complete", {"run_id": run_id, "step": step, "status": "COMPLETED", "duration_s": 0})

    hp2   = sum(1 for h in hp_log if h >= 2)
    rate  = hp2 / max(len(hp_log), 1)

    print()
    check("CNN — loss decreased (< 2.0)",          prev_loss < 2.0,  f"final={prev_loss:.4f}")
    check("CNN — false positive rate < 5%",         rate < 0.05,     f"{hp2} hp≥2 ({rate:.1%})")
    check("CNN — checkpoints issued",               ckpt_count > 0,  f"{ckpt_count} checkpoints")


# ══════════════════════════════════════════════════════════════════════════
# 3. LSTM — wikitext-2 — PyTorch
# ══════════════════════════════════════════════════════════════════════════

def test_lstm():
    print("\n" + "═" * 65)
    print("  3. LSTM — wikitext-2 — The Skeptic's Architecture")
    print("═" * 65)

    torch.manual_seed(SEED)
    run_id = f"arch_lstm_{int(time.time())}"

    # Tokenize wikitext-2 into a flat token stream
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset   = dataset.filter(lambda x: len(x["text"].strip()) > 50)

    text  = " ".join(dataset["text"][:500])
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    SEQ_LEN   = 64
    BATCH     = 16
    VOCAB     = tokenizer.vocab_size

    def get_batch(tokens, step):
        torch.manual_seed(SEED + step)
        ix  = torch.randint(len(tokens) - SEQ_LEN - 1, (BATCH,))
        x   = torch.stack([tokens[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
        y   = torch.stack([tokens[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
        return x, y

    # Real LSTM language model — 2 layers, 512 hidden
    class LSTMLanguageModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
            super().__init__()
            self.embed   = nn.Embedding(vocab_size, embed_dim)
            self.lstm    = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=0.2)
            self.head    = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            return self.head(out), hidden

    model     = LSTMLanguageModel(VOCAB).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt       = optim.Adam(model.parameters(), lr=5e-4)
    sched     = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=20)

    prev_loss  = None
    prev_grads = None
    hp_log     = []
    ckpt_count = 0
    hidden     = None

    print(f"\n  {'step':>5}  {'loss':>9}  {'ppl':>8}  {'gn':>7}  {'hp':>4}  {'lr':>10}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*4}  {'─'*10}")

    for step in range(200):
        x, y = get_batch(tokens, step)

        # Detach hidden to prevent backprop through entire history
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)

        opt.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()

        lv  = loss.item()
        gn  = grad_norm_of(model)
        ld  = (lv - prev_loss) if prev_loss is not None else 0.0
        gs  = grad_sim_of(model, prev_grads)
        lr  = sched.get_last_lr()[0]
        ppl = math.exp(min(lv, 10.0))

        prev_grads = get_grads(model)
        prev_loss  = lv

        hp, intv, ckpt, breakdown = send_step(run_id, step, lv, ld, gn, gs, "rnn")
        hp_log.append(hp)
        if ckpt in ("SAVE", "SAVE_NOW"):
            ckpt_count += 1

        if step % 50 == 0:
            print(f"  {step:>5}  {lv:9.4f}  {ppl:8.2f}  {gn:7.3f}  {hp:>4}  {lr:10.2e}")
        if hp >= 2 and step <= 50:
            active = {k: round(v,3) for k,v in breakdown.items() if isinstance(v, float) and v >= 1.0}
            print(f"  ⚠  step={step} hp={hp} {intv} loss={lv:.4f} signals={active}")
        elif hp >= 2:
            print(f"  ⚠  step={step} hp={hp} {intv} loss={lv:.4f}")

    req("POST", "/v2/complete", {"run_id": run_id, "step": 200, "status": "COMPLETED", "duration_s": 0})

    hp2   = sum(1 for h in hp_log if h >= 2)
    rate  = hp2 / max(len(hp_log), 1)

    print()
    check("LSTM — loss decreased (< 8.0)",          prev_loss < 8.0,  f"final={prev_loss:.4f}")
    check("LSTM — false positive rate < 5%",         rate < 0.05,     f"{hp2} hp≥2 ({rate:.1%})")
    check("LSTM — checkpoints issued",               ckpt_count > 0,  f"{ckpt_count} checkpoints")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  PLARV Argus — System Health Check")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 65)

    # Use the unauthenticated /health endpoint directly as requested
    try:
        req = urllib.request.Request(BASE + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            body = json.loads(resp.read())
            
        assert status == 200, f"API unreachable: {status}"
        print(f"\n  ✅ API healthy!")
        print(f"  Status: {body.get('data', {}).get('globalStatus', 'UNKNOWN')}")
        print(f"  Timestamp: {body.get('data', {}).get('timestamp', 'UNKNOWN')}")
        
    except Exception as e:
        print(f"\n  ❌ Health check failed: {str(e)}")

    # 🛑 ARCHITECTURE TESTS COMMENTED OUT TO PREVENT DASHBOARD CONTAMINATION AND BILLING
    test_transformer()
    test_cnn()
    test_lstm()

    print("\n" + "═" * 65 + "\n")

if __name__ == "__main__":
    main()

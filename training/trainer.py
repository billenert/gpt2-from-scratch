import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader=None,
        *,
        n_tokens: int,
        peak_lr: float = 3e-4,
        min_lr_ratio: float = 0.1,
        warmup_tokens: int | None = None,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        grad_clip: float = 1.0,
        log_every: int = 50,
        eval_every: int = 2000,
        eval_batches: int = 50,
        ckpt_every: int = 2000,
        ckpt_dir: str | Path = "checkpoints",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        # generation samples (set sample_prompts to enable)
        sample_prompts: list[str] | None = None,
        sample_every: int = 5000,
        sample_max_new_tokens: int = 40,
        sample_temperature: float = 0.8,
        sample_top_k: int = 40,
        n_ctx: int | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.n_tokens = n_tokens
        self.peak_lr = peak_lr
        self.min_lr_ratio = min_lr_ratio
        self.warmup_tokens = warmup_tokens if warmup_tokens is not None else max(n_tokens // 100, 1)
        self.grad_clip = grad_clip

        self.log_every = log_every
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        self.ckpt_every = ckpt_every
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.dtype = dtype
        self.use_autocast = dtype != torch.float32 and device != "cpu"

        self.optimizer = self._build_optimizer(weight_decay=weight_decay, betas=betas)

        self.sample_prompts = sample_prompts or []
        self.sample_every = sample_every
        self.sample_max_new_tokens = sample_max_new_tokens
        self.sample_temperature = sample_temperature
        self.sample_top_k = sample_top_k
        self.n_ctx = n_ctx

        if self.sample_prompts:
            if n_ctx is None:
                raise ValueError("n_ctx is required when sample_prompts is provided")
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("gpt2")
        else:
            self._tokenizer = None

        self.step = 0
        self.tokens_seen = 0


    def fit(self):
        self.model.train()
        pbar = tqdm(
            total=self.n_tokens,
            initial=self.tokens_seen,
            unit="tok",
            unit_scale=True,
            smoothing=0.05,
            dynamic_ncols=True,
        )

        try:
            for batch in self.train_loader:
                if self.tokens_seen >= self.n_tokens:
                    break

                tokens_before = self.tokens_seen
                loss, grad_norm = self._train_step(batch)
                self.step += 1
                pbar.update(self.tokens_seen - tokens_before)

                if self.step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{loss:.4f}",
                        lr=f"{lr:.2e}",
                        step=self.step,
                    )
                    if self._wandb_active():
                        wandb.log(
                            {
                                "train/loss": loss,
                                "train/lr": lr,
                                "train/grad_norm": grad_norm,
                                "train/tokens": self.tokens_seen,
                            },
                            step=self.step,
                        )

                if self.eval_every and self.val_loader and self.step % self.eval_every == 0:
                    val_loss = self.evaluate()
                    tqdm.write(f"[eval] step {self.step}  val_loss {val_loss:.4f}")
                    if self._wandb_active():
                        wandb.log({"val/loss": val_loss}, step=self.step)

                if self.step % self.ckpt_every == 0:
                    self.save_checkpoint()
                    tqdm.write(f"[ckpt] step {self.step}  tokens {self.tokens_seen / 1e6:.1f}M")

                if self.sample_prompts and self.sample_every and self.step % self.sample_every == 0:
                    samples = self.generate_samples()
                    for prompt, completion in samples:
                        tqdm.write(f"[sample] {prompt!r} → {completion!r}")
                    if self._wandb_active():
                        table = wandb.Table(columns=["prompt", "completion"])
                        for p, c in samples:
                            table.add_data(p, c)
                        wandb.log({"samples": table}, step=self.step)

        except KeyboardInterrupt:
            tqdm.write("interrupted — saving checkpoint")
        finally:
            pbar.close()

        self.save_checkpoint(name="final")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.val_loader):
            if i >= self.eval_batches:
                break
            inputs, targets = self._split(batch)
            with self._autocast():
                logits = self.model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def generate_samples(self) -> list[tuple[str, str]]:
        self.model.eval()
        results = []
        for prompt in self.sample_prompts:
            prompt_ids = self._tokenizer.encode_ordinary(prompt)
            tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            for _ in range(self.sample_max_new_tokens):
                ctx = tokens[:, -self.n_ctx:]
                with self._autocast():
                    logits = self.model(ctx)[:, -1, :]
                logits = logits / self.sample_temperature
                if self.sample_top_k:
                    k = min(self.sample_top_k, logits.shape[-1])
                    v, _ = logits.topk(k)
                    logits[logits < v[:, -1:]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
            completion = self._tokenizer.decode(tokens[0, len(prompt_ids):].tolist())
            results.append((prompt, completion))
        self.model.train()
        return results

    def save_checkpoint(self, name: str | None = None):
        path = self.ckpt_dir / (f"{name}.pt" if name else f"step_{self.step}.pt")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "tokens_seen": self.tokens_seen,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]

    def _train_step(self, batch: torch.Tensor) -> tuple[float, float]:
        lr = self._lr_for(self.tokens_seen)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        inputs, targets = self._split(batch)

        with self._autocast():
            logits = self.model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ returns the pre-clip total norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.tokens_seen += inputs.numel()
        return loss.item(), grad_norm.item()

    def _wandb_active(self) -> bool:
        return wandb is not None and wandb.run is not None

    def _split(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    def _autocast(self):
        if self.use_autocast:
            return torch.amp.autocast(device_type=self.device, dtype=self.dtype)
        return torch.amp.autocast(device_type="cpu", enabled=False)

    def _build_optimizer(self, *, weight_decay: float, betas: tuple[float, float]):
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        param_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=self.peak_lr, betas=betas)

    def _lr_for(self, tokens_seen: int) -> float:
        if tokens_seen < self.warmup_tokens:
            return self.peak_lr * (tokens_seen + 1) / self.warmup_tokens
        progress = (tokens_seen - self.warmup_tokens) / max(self.n_tokens - self.warmup_tokens, 1)
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
        return self.peak_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * coeff)

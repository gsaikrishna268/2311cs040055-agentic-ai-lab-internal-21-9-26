"""
=============================================================================================
MALLA REDDY UNIVERSITY (MRU) - DEPARTMENT OF CYBER SECURITY
Course: (MR23-1CS0436) APPLIED AGENTIC AI (B.Tech R-23)
Lab Experiment 10: Fine-Tuning for Domain Adaptation
Objective: Train and evaluate a specialized LLM for cybersecurity vulnerability & threat analysis.
=============================================================================================
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output encoding for cross-platform compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import track
    from rich import box
    console = Console(force_terminal=True)
except ImportError:
    console = None


# =============================================================================================
# 1. DOMAIN-SPECIFIC DATASET: CYBERSECURITY THREAT INTELLIGENCE & CVE REMEDIATION
# =============================================================================================
CYBER_DOMAIN_CORPUS = [
    {
        "prompt": "[VULN_QUERY] CVE-2024-3094 SSH XZ Utils backdoor discovered in liblzma upstream repository.",
        "response": "[ANALYSIS] Threat: Supply-chain backdoor targeting sshd auth. CVSS: 10.0 Critical. Mitigation: Downgrade xz-utils to 5.4.6 and rebuild OpenSSH binaries immediately."
    },
    {
        "prompt": "[VULN_QUERY] CVE-2023-38606 Kernel privilege escalation exploit bypassing Page Protection Layer.",
        "response": "[ANALYSIS] Threat: Hardware memory-mapped I/O vulnerability. CVSS: 8.8 High. Mitigation: Apply vendor kernel patches; restrict kernel extension loading via Mobile Device Management."
    },
    {
        "prompt": "[VULN_QUERY] CVE-2024-21887 Command injection vulnerability in Ivanti Connect Secure Web interface.",
        "response": "[ANALYSIS] Threat: Remote code execution without prior authentication. CVSS: 9.1 Critical. Mitigation: Apply XML mitigation package and conduct active triage for web shells."
    },
    {
        "prompt": "[VULN_QUERY] CVE-2023-4863 Heap buffer overflow in WebP lossless decoding library libwebp.",
        "response": "[ANALYSIS] Threat: Memory corruption during Huffman table parsing. CVSS: 8.8 High. Mitigation: Update Chromium and libwebp packages to patched versions across all rendering engines."
    },
    {
        "prompt": "[VULN_QUERY] CVE-2024-4577 CGI parameter injection in PHP-CGI Windows deployments.",
        "response": "[ANALYSIS] Threat: Best-fit character encoding bypass leading to RCE. CVSS: 9.8 Critical. Mitigation: Update PHP to 8.1.29+ and disable legacy Action directive mappings in Apache."
    },
    {
        "prompt": "[VULN_QUERY] CVE-2023-34362 SQL Injection in MOVEit Transfer web application interface.",
        "response": "[ANALYSIS] Threat: Pre-auth data exfiltration via crafted HTTP payloads. CVSS: 9.8 Critical. Mitigation: Disable external HTTP/HTTPS traffic to MOVEit and patch database drivers."
    }
]

# Validation / Evaluation Domain Queries
CYBER_EVAL_QUERIES = [
    {
        "prompt": "[VULN_QUERY] CVE-2024-3094 SSH XZ Utils backdoor discovered in liblzma upstream repository.",
        "expected": "Mitigation: Downgrade xz-utils to 5.4.6 and rebuild OpenSSH binaries immediately."
    },
    {
        "prompt": "[VULN_QUERY] CVE-2024-21887 Command injection vulnerability in Ivanti Connect Secure Web interface.",
        "expected": "Mitigation: Apply XML mitigation package and conduct active triage for web shells."
    }
]


# =============================================================================================
# 2. TOKENIZATION & VOCABULARY MANAGEMENT
# =============================================================================================
class CyberDomainTokenizer:
    """Character & Subword Domain Tokenizer for fast, deterministic embedding."""

    def __init__(self):
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        self.vocab = {}
        self.inv_vocab = {}
        self._build_vocab()

    def _build_vocab(self):
        chars = sorted(list(set(
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:-_/[]()+='\",;!?#\n\t"
        )))
        all_tokens = self.special_tokens + chars
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.inv_vocab = {idx: tok for idx, tok in enumerate(all_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        tokens = [self.vocab.get(ch, self.vocab[self.unk_token]) for ch in text]
        tokens.append(self.vocab[self.eos_token])
        if len(tokens) < max_len:
            tokens = tokens + [self.vocab[self.pad_token]] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, token_ids: List[int]) -> str:
        res = []
        for tid in token_ids:
            if tid == self.vocab[self.pad_token] or tid == self.vocab[self.eos_token]:
                break
            res.append(self.inv_vocab.get(tid, ""))
        return "".join(res)


# =============================================================================================
# 3. DOMAIN ADAPTATION TRANSFORMER MODEL ARCHITECTURE (PyTorch)
# =============================================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CyberDomainLM(nn.Module):
    """
    Lightweight Causal Language Model for Domain Adaptation Fine-Tuning.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_layers: int = 4, num_heads: int = 4, max_seq_len: int = 256):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, max_seq_len) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, Any]:
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, tokenizer: CyberDomainTokenizer, prompt: str, max_new_tokens: int = 60) -> str:
        """Autoregressive generation for domain query inference."""
        self.eval()
        encoded = tokenizer.encode(prompt, max_len=180)
        # trim padding
        input_ids = [t for t in encoded if t != tokenizer.vocab[tokenizer.pad_token] and t != tokenizer.vocab[tokenizer.eos_token]]
        x = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_crop = x[:, -self.max_seq_len:]
                logits, _ = self(x_crop)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                if next_token.item() == tokenizer.vocab[tokenizer.eos_token]:
                    break
                x = torch.cat([x, next_token], dim=1)

        generated_ids = x[0].tolist()[len(input_ids):]
        return tokenizer.decode(generated_ids)


# =============================================================================================
# 4. FINE-TUNING AND EVALUATION PIPELINE
# =============================================================================================
class DomainAdaptationTrainer:
    """
    Manages Domain Dataset tokenization, Pre-adaptation evaluation,
    Fine-Tuning optimization loop, and Post-adaptation evaluation.
    """

    def __init__(self):
        self.tokenizer = CyberDomainTokenizer()
        self.model = CyberDomainLM(vocab_size=self.tokenizer.vocab_size)
        self.dataset_tensors = self._prepare_data()

    def _prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_list = []
        target_list = []
        for item in CYBER_DOMAIN_CORPUS:
            full_text = f"{item['prompt']} {item['response']}"
            tokens = self.tokenizer.encode(full_text, max_len=200)
            # autoregressive target is shifted by 1
            input_list.append(tokens[:-1])
            target_list.append(tokens[1:])

        return torch.tensor(input_list, dtype=torch.long), torch.tensor(target_list, dtype=torch.long)

    def evaluate_perplexity(self) -> Tuple[float, float]:
        """Calculates loss and Perplexity (PPL = exp(loss)) on the domain corpus."""
        self.model.eval()
        x, y = self.dataset_tensors
        with torch.no_grad():
            _, loss = self.model(x, y)
            val_loss = loss.item()
            perplexity = math.exp(min(val_loss, 20.0))  # numerical stability
        return round(val_loss, 4), round(perplexity, 2)

    def train_domain_adaptation(self, epochs: int = 40, lr: float = 0.002) -> List[Dict[str, Any]]:
        """Fine-tunes the model on domain-specific vocabulary and incident responses."""
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        x, y = self.dataset_tensors

        history = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            _, loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                ppl = math.exp(min(loss.item(), 20.0))
                history.append({"epoch": epoch, "loss": round(loss.item(), 4), "perplexity": round(ppl, 2)})

        return history


# =============================================================================================
# MAIN LAB EXECUTION & COMPARATIVE REPORTING
# =============================================================================================
def main():
    print("=" * 80)
    print("🚀 EXPERIMENT 10: FINE-TUNING LLM FOR CYBERSECURITY DOMAIN ADAPTATION")
    print("=" * 80)

    trainer = DomainAdaptationTrainer()

    # Step 1: Pre-Fine-Tuning (Zero-Shot / Base) Evaluation
    print("\n[Stage 1/4] Evaluating Base Pre-trained Model on Domain Corpus...")
    pre_loss, pre_ppl = trainer.evaluate_perplexity()
    test_query = CYBER_EVAL_QUERIES[0]["prompt"]
    pre_generation = trainer.model.generate(trainer.tokenizer, test_query, max_new_tokens=50)

    # Step 2: Fine-Tuning Execution
    print("[Stage 2/4] Executing Parameter-Efficient Fine-Tuning Loop (Epochs: 40, AdamW)...")
    start_time = time.time()
    train_history = trainer.train_domain_adaptation(epochs=40, lr=0.003)
    train_duration = round(time.time() - start_time, 2)

    # Step 3: Post-Fine-Tuning Evaluation
    print("[Stage 3/4] Evaluating Adapted Specialized Model on Domain Corpus...")
    post_loss, post_ppl = trainer.evaluate_perplexity()
    post_generation = trainer.model.generate(trainer.tokenizer, test_query, max_new_tokens=65)

    # Step 4: Display Results
    if console:
        console.print(Panel.fit("[bold green]Lab Experiment 10: Fine-Tuning for Domain Adaptation Completed[/bold green]", border_style="green"))

        # Quantitative Metrics Table
        metrics_table = Table(title="Domain Adaptation Quantitative Performance", box=box.ROUNDED)
        metrics_table.add_column("Evaluation Metric", style="cyan")
        metrics_table.add_column("Pre-Trained Base Model", style="red")
        metrics_table.add_column("Fine-Tuned Specialized Model", style="green")
        metrics_table.add_column("Improvement Delta", style="bold yellow")

        loss_delta = round(pre_loss - post_loss, 4)
        ppl_delta = round(((pre_ppl - post_ppl) / pre_ppl) * 100, 2)

        metrics_table.add_row("Domain Cross-Entropy Loss", f"{pre_loss}", f"{post_loss}", f"-{loss_delta} (Lower is better)")
        metrics_table.add_row("Corpus Perplexity (PPL)", f"{pre_ppl}", f"{post_ppl}", f"{ppl_delta}% PPL Reduction")
        metrics_table.add_row("Training Execution Time", "N/A", f"{train_duration}s", "Optimized")
        console.print(metrics_table)

        # Epoch Loss Convergence Table
        epoch_table = Table(title="Training Convergence History (Loss & Perplexity)", box=box.ROUNDED)
        epoch_table.add_column("Epoch", style="cyan")
        epoch_table.add_column("Training Loss", style="magenta")
        epoch_table.add_column("Perplexity (PPL)", style="yellow")
        for h in train_history:
            epoch_table.add_row(str(h["epoch"]), str(h["loss"]), str(h["perplexity"]))
        console.print(epoch_table)

        # Qualitative Output Comparison
        comp_md = (
            f"**Test Input Vulnerability Query:**\n`{test_query}`\n\n"
            f"**1. Pre-Trained Model Generation (Untuned / Random Base):**\n"
            f"> *{pre_generation if pre_generation.strip() else '[Incoherent / Non-domain random tokens]'}*\n\n"
            f"**2. Fine-Tuned Specialized Agent Generation (Adapted):**\n"
            f"> **{post_generation}**\n\n"
            f"**Target Expert Ground Truth:**\n"
            f"`{CYBER_EVAL_QUERIES[0]['expected']}`"
        )
        console.print(Panel(comp_md, title="[bold cyan]Qualitative Domain Adaptation Inference Comparison[/bold cyan]", border_style="cyan"))

    else:
        print("\n" + "=" * 60)
        print("DOMAIN ADAPTATION EVALUATION METRICS")
        print(f"Pre-Tuning Loss: {pre_loss} | PPL: {pre_ppl}")
        print(f"Post-Tuning Loss: {post_loss} | PPL: {post_ppl}")
        print("=" * 60)
        print(f"Query: {test_query}")
        print(f"Post-Tuning Output: {post_generation}")


if __name__ == "__main__":
    main()

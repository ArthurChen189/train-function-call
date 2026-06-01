"""SFT with LoRA on Qwen3-0.6B using TRL's SFTTrainer.

Why TRL/PEFT instead of Unsloth:
- One fewer moving dep; transformer 4.50+ + peft 0.14 + trl 0.21 is rock-solid.
- Qwen3-0.6B in bf16 with a small LoRA fits in ~6GB VRAM, so Unsloth's
  4-bit + custom kernels aren't necessary at this scale.
- Easy to swap in Unsloth later by replacing `AutoModelForCausalLM.from_pretrained`
  with `FastLanguageModel.from_pretrained`.

Trains only on assistant tokens via TRL's `assistant_only_loss=True`, so the
gradient never tries to memorise the system prompt or tool outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .data import build_sft_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bsz", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    # APIGen-MT trajectories are long (shared system prompt alone is ~3.6k
    # tokens); 6144 keeps the great majority of full multi-turn trajectories.
    p.add_argument("--max-seq-len", type=int, default=6144)
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sft] loading tokenizer + model from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto",
    )
    model.config.use_cache = False

    print(f"[sft] loading APIGen-MT-5k SFT data n_train={args.n_train} "
          f"n_eval={args.n_eval} max_seq_len={args.max_seq_len}")
    train_ds = build_sft_dataset(
        args.n_train, seed=args.seed, split="train",
        tokenizer=tokenizer, max_tokens=args.max_seq_len,
    )
    eval_ds = build_sft_dataset(
        args.n_eval, seed=args.seed, split="eval",
        tokenizer=tokenizer, max_tokens=args.max_seq_len,
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        # Long sequences -> trade compute for memory so 6k-token multi-turn
        # trajectories fit on a 24GB GPU.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=True,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    print("[sft] starting training")
    trainer.train()

    final = out_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[sft] saved adapter to {final}")


if __name__ == "__main__":
    main()

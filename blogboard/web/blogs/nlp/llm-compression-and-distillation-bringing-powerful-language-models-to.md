# LLM Compression and Distillation: Bringing Powerful Language Models to the Edge  

*By [Your Name], Senior Technical Writer – NLP & Edge‑AI*  

---

## Table of Contents
1. [Why Edge‑Ready LLMs Matter](#why-edge-ready-llms-matter)  
2. [Model Pruning and Quantization Techniques](#model-pruning-and-quantization-techniques)  
   - 2.1 Structured vs. Unstructured Pruning  
   - 2.2 Post‑Training Quantization (PTQ)  
   - 2.3 Quantization‑Aware Training (QAT)  
   - 2.4 Practical Code Snippets (🤖 bitsandbytes & 🤗 optimum)  
3. [Knowledge Distillation Strategies for LLMs](#knowledge-distillation-strategies-for-llms)  
   - 3.1 Classic Teacher‑Student Framework  
   - 3.2 Data‑Free Distillation  
   - 3.3 Multi‑Task & Multi‑Teacher Distillation  
   - 3.4 Sample Distillation Pipeline (🤗 transformers)  
4. [Efficient Inference on Edge Devices](#efficient-inference-on-edge-devices)  
   - 4.1 Runtime Choices (ONNX Runtime, TensorRT, TVM, vLLM)  
   - 4.2 Memory‑Mapping & Off‑loading Strategies  
   - 4.3 Latency‑Optimized Prompt Formatting  
5. [Real‑World Deployment Case Studies](#real-world-deployment-case-studies)  
   - 5.1 Smart‑Home Hub (completed)  
   - 5.2 Mobile Personal Assistant (new)  
   - 5.3 Trade‑offs & Lessons Learned  
6. [Deployment Checklist for Edge LLMs](#deployment-checklist-for-edge-llms)  
7. [Conclusion & Actionable Takeaways](#conclusion--actionable-takeaways)  
8. [References](#references)  

---

## Why Edge‑Ready LLMs Matter

Large language models (LLMs) have revolutionized natural language understanding, but their **billions of parameters** and **high memory bandwidth** make them unsuitable for many on‑device scenarios:

| Edge Constraint | Typical Requirement | Why LLMs Struggle |
|-----------------|---------------------|-------------------|
| **Compute** | < 2 TOPS (CPU/GPU) | Transformer ops are matrix‑heavy |
| **Memory** | ≤ 2 GB RAM (incl. OS) | 7B‑parameter model ≈ 14 GB FP16 |
| **Power** | < 5 W (battery) | Continuous inference drains batteries |
| **Latency** | < 200 ms (real‑time) | Large beam search adds overhead |

Compressing and distilling LLMs enables **privacy‑preserving on‑device AI**, **offline operation**, and **lower operational costs**—key for IoT, wearables, and remote deployments.

---

## Model Pruning and Quantization Techniques

### 2.1 Structured vs. Unstructured Pruning  

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| **Unstructured (weight‑level)** | Zero‑out individual weights based on magnitude or sensitivity. | Highest sparsity → biggest size reduction. | Irregular sparsity hurts hardware acceleration; often needs a sparse kernel. |
| **Structured (head/row/column)** | Remove entire attention heads, MLP columns, or feed‑forward neurons. | Keeps dense matrix shapes → compatible with existing BLAS kernels. | Less aggressive compression; may require more fine‑tuning. |

**Practical tip:** Start with **structured pruning of attention heads** (e.g., prune heads with low attention entropy) before moving to fine‑grained unstructured sparsity.

```python
from transformers import AutoModelForCausalLM
from torch.nn.utils import prune

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Example: prune 30 % of the feed‑forward neurons in each MLP layer
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "mlp" in name:
        prune.ln_structured(module, name="weight", amount=0.3, n=2)  # n=2 → column pruning
```

> **Note:** After pruning, always **re‑initialize the optimizer** and run a short fine‑tuning pass (≈ 1 epoch on a domain‑specific corpus) to recover accuracy.

### 2.2 Post‑Training Quantization (PTQ)

PTQ converts FP32/FP16 weights to lower‑precision integers **without additional training**. The most common formats are:

| Format | Bit‑width | Typical Speed‑up | Accuracy Impact |
|--------|-----------|------------------|-----------------|
| **INT8** | 8 bits | 2–3× on CPUs, 4–5× on GPUs | < 1 % BLEU loss for many tasks |
| **INT4** | 4 bits (via **bitsandbytes**) | 5–7× on modern GPUs | 1–3 % drop; acceptable for classification, less for generation |
| **FP8** (NVIDIA) | 8 bits (floating) | 2–3× on Tensor Cores | Near‑FP16 accuracy for many LLMs |

#### Corrected bitsandbytes PTQ Example  

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4‑bit quantization with NF4 (normal‑float‑4) and double‑quantization enabled
quantized_model = bnb.nn.Int8Params.convert_to_int8(
    model,
    dtype=torch.float16,
    quant_type="nf4",          # <-- correct argument name
    double_quant=True,         # <-- enable double quantization
    module_name="model"
)

# Verify a forward pass
input_ids = tokenizer("Edge AI is", return_tensors="pt").input_ids.to(quantized_model.device)
with torch.no_grad():
    logits = quantized_model(input_ids).logits
print(logits.shape)
```

> **Why `dtype=torch.float16`?** Bitsandbytes stores the **scaling factors** in FP16, which preserves dynamic range while keeping the weight matrix in 4‑bit.

### 2.3 Quantization‑Aware Training (QAT)

QAT inserts **fake quantization nodes** during training so the model learns to compensate for the reduced precision. This typically yields **higher fidelity** than PTQ, especially for **generation tasks**.

```python
from optimum.intel import IncQuantizer, IncQuantizationConfig
from transformers import Trainer, TrainingArguments

# Load a pre‑quantized checkpoint (e.g., 8‑bit)
quantizer = IncQuantizer.from_pretrained(model_name, quantization_config=IncQuantizationConfig(
    weight_dtype="int8", activation_dtype="int8", per_channel=True))

quantized_model = quantizer.quantize_model(model)

training_args = TrainingArguments(
    output_dir="./qat_llama7b",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=1,
    fp16=True,
)

trainer = Trainer(
    model=quantized_model,
    args=training_args,
    train_dataset=small_dataset,
)

trainer.train()
```

**Tip:** Use a **small, high‑quality calibration set** (≈ 500 sentences) for QAT; this dramatically reduces the risk of catastrophic forgetting.

### 2.4 Practical Code Snippets (🤗 optimum)

Optimum provides a unified API for **ONNX Runtime**, **TensorRT**, and **OpenVINO** quantization. Below is a minimal PTQ pipeline for a 7B model targeting **ONNX Runtime** on a Raspberry Pi (ARM v8).

```python
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Export to ONNX (FP16)
ort_model = ORTModelForCausalLM.from_pretrained(model_id, export=True, fp16=True)

# PTQ to INT8 using static calibration
quantizer = ORTQuantizer.from_pretrained(ort_model)
calibration_dataset = ["Edge devices need fast NLP.", "Quantization reduces memory."]

quantizer.quantize(
    save_dir="./llama2-7b-onnx-int8",
    calibration_dataset=calibration_dataset,
    calibrate_method="minmax",   # or "entropy"
    per_channel=True,
)

# Load the quantized model for inference
quantized_ort = ORTModelForCausalLM.from_pretrained("./llama2-7b-onnx-int8")
input_ids = tokenizer("Explain quantization", return_tensors="pt").input_ids
outputs = quantized_ort.generate(input_ids, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Knowledge Distillation Strategies for LLMs

### 3.1 Classic Teacher‑Student Framework  

The **teacher** is a large, high‑capacity model (e.g., Llama‑2‑70B). The **student** is a smaller architecture (e.g., 1.5 B) trained to match the teacher’s **logits**, **hidden states**, or **attention maps**.

**Loss formulation** (cross‑entropy + KL divergence):

\[
\mathcal{L} = \alpha \cdot \text{CE}(y, \hat{y}) + \beta \cdot \text{KL}\big(\sigma(z_T / \tau) \,\|\, \sigma(z_S / \tau)\big)
\]

- \(\tau\) = temperature (commonly 2–5).  
- \(\alpha, \beta\) balance ground‑truth supervision and distillation.

### 3.2 Data‑Free Distillation  

When proprietary data cannot be shared, **synthetic data** generated by the teacher (or a separate language model) can be used.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

def generate_synthetic(batch_size=8, max_len=128):
    prompts = ["Explain", "Summarize", "Define", "Compare"] * (batch_size // 4)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(teacher.device)
    with torch.no_grad():
        outputs = teacher.generate(**inputs, max_length=max_len, do_sample=True, top_p=0.95)
    return outputs

synthetic_ids = generate_synthetic()
```

The synthetic tokens become the **distillation dataset** for the student.

### 3.3 Multi‑Task & Multi‑Teacher Distillation  

- **Multi‑Task:** Train a single student to handle **instruction following**, **code generation**, and **retrieval‑augmented QA** simultaneously.  
- **Multi‑Teacher:** Blend logits from several specialized teachers (e.g., a code‑LLM + a dialogue‑LLM) using **weighted averaging**.

```python
# Example: weighted logit blending
teacher_a = AutoModelForCausalLM.from_pretrained("code-llama-7b")
teacher_b = AutoModelForCausalLM.from_pretrained("dialogue-llama-7b")

def blended_logits(input_ids, alpha=0.6):
    logits_a = teacher_a(input_ids).logits
    logits_b = teacher_b(input_ids).logits
    return alpha * logits_a + (1 - alpha) * logits_b
```

### 3.4 Sample Distillation Pipeline (🤗 transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch.nn.functional as F

teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
student = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-3b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(**inputs)
        student_out = model(**inputs)
        loss_ce = F.cross_entropy(
            student_out.logits.view(-1, student_out.logits.size(-1)),
            inputs["labels"].view(-1),
            ignore_index=-100,
        )
        loss
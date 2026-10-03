# Self‑Supervised Learning for Audio & Speech Representation  
*A practical, end‑to‑end tutorial*

---

## Introduction  

Self‑supervised learning (SSL) has reshaped how we build audio and speech systems. By extracting useful representations from raw waveforms without any human‑annotated labels, SSL enables **massive pre‑training** on publicly available audio corpora and **efficient fine‑tuning** on downstream tasks such as automatic speech recognition (ASR), speaker identification, and emotion recognition.  

This tutorial walks through:

1. **Core SSL concepts** – Contrastive Predictive Coding (CPC) and the wav2vec family.  
2. **Multimodal audio‑visual self‑supervision** – How visual cues boost acoustic representations.  
3. **Large‑scale pretrained models** – What’s available off‑the‑shelf and how they perform on common downstream tasks.  
4. **Efficient fine‑tuning & edge deployment** – Strategies for low‑resource hardware and a complete code walk‑through that ends with a TensorFlow Lite (TFLite) export.  

By the end you will have a working pipeline that fine‑tunes a wav2vec 2.0 model on a small speech dataset and produces a lightweight TFLite file ready for on‑device inference.

---  

## 1. Foundations of Audio SSL  

### 1.1 Contrastive Predictive Coding (CPC)  

CPC (Oord *et al.*, 2018) introduced the idea of **predicting future latent representations** while maximizing a contrastive loss. The workflow is:

1. **Encoder** → maps raw waveform \(x_t\) to a latent sequence \(z_t\).  
2. **Autoregressive model** → aggregates context up to time \(t\) into \(c_t\).  
3. **Prediction** → \(c_t\) is used to predict \(z_{t+k}\) for several future steps \(k\).  
4. **Contrastive loss** → the correct future latent is distinguished from a set of negative samples drawn from other time steps or other utterances.

The loss encourages the encoder to retain information that is **predictable over time**, which aligns well with phonetic and prosodic structure in speech.

### 1.2 The wav2vec Family  

| Model | Year | Core Idea | Key Improvements |
|-------|------|-----------|-------------------|
| **wav2vec** | 2019 (Schneider *et al.*) | CPC on raw audio, 1‑D convolutional encoder + quantization of latent vectors. | Introduced **discrete quantization** that enables a contrastive loss over a finite codebook. |
| **wav2vec 2.0** | 2020 (Baevski *et al.*) | Masked acoustic modeling (MAM) – randomly mask latent frames and predict them from the unmasked context. | **Transformer** context network, **large‑scale pre‑training** on 60 k h of Librispeech + LibriVox, state‑of‑the‑art ASR without any labeled data. |
| **HuBERT** | 2021 (Hsu *et al.*) | Iterative clustering: first generate pseudo‑labels by k‑means on MFCCs, then train a masked model; repeat with model‑derived clusters. | Better **phoneme‑level alignment**, strong performance on multilingual speech tasks. |
| **Data2Vec** | 2022 (Baevski *et al.*) | Unified SSL across modalities – the target is the **latent representation** of a teacher network, not discrete tokens. | **Cross‑modal flexibility**, competitive results on speech and audio classification. |
| **WavLM** | 2022 (Chen *et al.*) | Adds **utterance‑level mixing** and **speaker‑aware masking** to wav2vec 2.0. | Improves robustness to noise and speaker variation; excels on speaker‑verification and diarization. |

All models share a **masked prediction** paradigm: a portion of the latent sequence is hidden, and the network must reconstruct it using the surrounding context. This yields representations that capture both **local acoustic detail** (important for phoneme discrimination) and **long‑range dependencies** (useful for speaker and emotion cues).

---  

## 2. Multimodal Audio‑Visual Self‑Supervision  

Visual information—lip movements, facial expressions, scene context—provides a powerful supervisory signal for audio. Two representative approaches are highlighted below.

### 2.1 AV‑Hubert  

*AV‑Hubert* (Shi *et al.*, 2022) extends HuBERT by **jointly masking audio and video frames** and predicting the masked modalities from the unmasked counterpart. The architecture consists of:

* **Audio encoder** – same convolution + Transformer as HuBERT.  
* **Video encoder** – 3‑D CNN that processes cropped mouth regions.  
* **Cross‑modal Transformer** – fuses audio and visual tokens via self‑attention.  

Training on **LRS3‑TED** (audio‑visual speech) yields a model that outperforms audio‑only SSL on lip‑reading and noisy ASR, demonstrating that visual cues help the model learn **speaker‑independent phonetic representations**.

### 2.2 Contrastive Audio‑Visual Pre‑training (CAVP)  

CAVP (Morgado *et al.*, 2020) uses a **dual‑branch contrastive loss**: audio embeddings are pulled together with temporally aligned video embeddings while being pushed apart from mismatched pairs. This simple formulation scales to **hundreds of thousands of video clips** and produces embeddings that are effective for **audio‑visual event detection** and **speech separation**.

### 2.3 Takeaway  

When a visual stream is available, **mask‑and‑predict** across modalities or **contrastive alignment** can dramatically improve robustness to background noise and speaker variability. For pure audio pipelines, the same ideas can be simulated with **synthetic augmentations** (e.g., adding reverberation, pitch shift) that act as “pseudo‑visual” signals.

---  

## 3. Large‑Scale Pre‑trained Audio Models for Downstream Tasks  

Below we summarize the most widely used pretrained checkpoints and their typical performance on three downstream benchmarks.

| Model | Pre‑training Corpus | ASR (LibriSpeech test‑clean) | Speaker ID (VoxCeleb‑1) | Emotion Recognition (IEMOCAP) |
|-------|--------------------|------------------------------|--------------------------|--------------------------------|
| wav2vec 2.0 Base | 960 h Librispeech | 2.9 % WER (fine‑tuned) | 94 % top‑1 | 71 % accuracy (linear probe) |
| wav2vec 2.0 Large | 60 k h LibriVox | 1.8 % WER | 96 % top‑1 | 73 % accuracy |
| HuBERT Base | 960 h Librispeech + 60 k h LibriVox | 2.5 % WER | 95 % top‑1 | 72 % accuracy |
| WavLM Base+ | 94 k h (including noisy speech) | 2.2 % WER | 96 % top‑1 | 74 % accuracy |
| Data2Vec Audio Base | 60 k h | 2.4 % WER | 94 % top‑1 | 70 % accuracy |
| AV‑Hubert Large | 1 M h audio‑visual (LRS3) | 2.0 % WER (audio‑only) | 96 % top‑1 | 78 % accuracy (audio‑visual) |

*Numbers are representative of published results; actual performance may vary with data preprocessing and fine‑tuning hyper‑parameters.*

### 3.1 Why One Model Fits Many Tasks  

* **Shared acoustic space** – The same latent vector encodes phonetics, speaker characteristics, and prosody.  
* **Linear probing** – A simple logistic regression on frozen embeddings often yields strong baselines, allowing rapid experimentation.  
* **Task‑specific heads** – Adding a small classification or CTC head tailors the representation without retraining the whole encoder.

---  

## 4. Efficient Fine‑Tuning & Edge Deployment  

### 4.1 Parameter‑Efficient Adaptation  

| Technique | Description | Typical Speed‑up |
|-----------|-------------|------------------|
| **Adapter modules** (Houlsby *et al.*, 2019) | Insert tiny bottleneck layers (e.g., 64 hidden units) after each Transformer block; only adapters are trained. | 5–10× fewer trainable parameters |
| **Layer‑wise learning rate decay** | Apply a higher LR to the top layers, lower LR to the bottom. | Faster convergence, less over‑fitting |
| **Quantization‑aware training (QAT)** | Simulate 8‑bit integer arithmetic during fine‑tuning. | Up to 4× reduction in model size with <1 % accuracy loss |
| **Knowledge distillation** | Train a smaller student model to mimic the logits of a large teacher. | Model size ↓ 2–3×, latency ↓ 30 % |

These methods let you adapt a 300 M‑parameter wav2vec 2.0 Large model to a niche domain (e.g., medical dictation) while keeping the on‑device footprint modest.

### 4.2 From PyTorch → TensorFlow Lite  

The most common workflow is:

1. **Fine‑tune in PyTorch** (using 🤗 Transformers).  
2. **Export to ONNX** (`torch.onnx.export`).  
3. **Convert ONNX → TensorFlow** (`tf2onnx.convert`).  
4. **Apply post‑training quantization** (`tf.lite.TFLiteConverter`).  

The code snippet below demonstrates the full pipeline on a small custom ASR dataset (e.g., a few hours of command‑type speech).  

---  

## 5. Hands‑On: Fine‑Tuning wav2vec 2.0 and Exporting to TFLite  

### 5.1 Setup  

```bash
pip install torch torchvision torchaudio transformers datasets soundfile librosa tqdm tensorflow==2.13 onnx tf2onnx
```

### 5.2 Load Data  

```python
from datasets import load_dataset, load_metric
import torchaudio

# Example: CommonVoice English subset (small split for demo)
dataset = load_dataset("common_voice", "en", split="train[:5%]")
testset = load_dataset("common_voice", "en", split="validation[:5%]")

def preprocess(batch):
    speech_array, sr = torchaudio.load(batch["path"])
    # Resample to 16 kHz if needed
    if sr != 16000:
        speech_array = torchaudio.functional.resample(speech_array, sr, 16000)
    batch["input_values"] = speech_array.squeeze().numpy()
    batch["labels"] = batch["sentence"]
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
testset = testset.map(preprocess, remove_columns=testset.column_names)
```

### 5.3 Tokenizer & Model  

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    gradient_checkpointing=True,   # saves GPU memory
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
```

### 5.4 Data Collator  

```python
def data_collator(batch):
    input_values = [b["input_values"] for b in batch]
    labels = processor(text=[b["labels"] for b in batch], return_tensors="pt", padding=True).input_ids
    # Pad audio to longest sample in batch
    inputs = processor.feature_extractor(
        raw_speech=input_values,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.attention_mask,
        "labels": labels,
    }
```

### 5.5 Training Arguments  

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    learning_rate=3e-4,
    warmup_steps=500,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    load
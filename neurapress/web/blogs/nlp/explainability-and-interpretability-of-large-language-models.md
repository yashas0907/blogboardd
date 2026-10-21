# Explainability and Interpretability of Large Language Models  

*Large language models (LLMs) have transformed natural‑language processing, yet their opaque decision‑making raises practical, ethical, and scientific concerns. This tutorial surveys the most influential techniques for probing internal representations, attributing model outputs, performing causal and counterfactual analyses, and evaluating interpretability with systematic benchmarks.*  

---  

## 1. Introduction  

LLMs such as GPT‑4, PaLM, and LLaMA contain billions of parameters and are trained on massive text corpora. Their impressive capabilities come at the cost of **opacity**: a user sees only the final token sequence, while the internal dynamics that produced it remain hidden.  

Interpretability research aims to answer questions such as:

* **What linguistic or world knowledge is encoded in each layer?**  
* **Which input tokens or sub‑structures drive a particular prediction?**  
* **How would the model’s behavior change under controlled interventions?**  

Answering these questions is essential for debugging, bias mitigation, safety assurance, and scientific insight into language understanding. The field has converged around four complementary pillars:

1. **Probing and diagnostic techniques** that query hidden states.  
2. **Attribution and saliency methods** that map outputs back to inputs.  
3. **Causal and counterfactual analyses** that test “what‑if” scenarios.  
4. **Benchmarks and metrics** that quantify interpretability quality.  

The following sections provide a deep dive into each pillar, highlighting seminal works, methodological nuances, and practical recommendations for researchers and engineers.

---  

## 2. Probing and Diagnostic Techniques for Internal Representations  

### 2.1 What is Probing?  

**Probing** treats a frozen LLM as a feature extractor. A lightweight classifier (the *probe*) is trained on the hidden states to predict a linguistic property (e.g., part‑of‑speech, syntactic depth, coreference). If the probe succeeds, the property is said to be *encoded* in that layer.  

Key considerations:  

* **Control for probe capacity** – a powerful probe may learn the task itself rather than extracting information.  
* **Selectivity vs. sensitivity** – a probe should be sensitive enough to detect the signal but selective enough to avoid over‑fitting.  

### 2.2 Representative Probing Studies  

| Study | LLMs Examined | Probed Phenomena | Notable Findings |
|-------|---------------|------------------|------------------|
| Tenney *et al.* (2019) – “What Does BERT Look At?” | BERT, RoBERTa | POS, syntactic constituency, semantic roles | Lower layers capture surface syntax; higher layers encode semantic information. |
| Liu *et al.* (2021) – “Linguistic Knowledge and Transferability of Contextual Representations” | GPT‑2, T5 | Morphology, syntax, coreference | Transferability peaks in middle layers; fine‑tuning reshapes representations. |
| Rogers *et al.* (2020) – “A Primer on Neural Network Probing” | Various Transformers | Hierarchical structure, factual knowledge | Introduced *control tasks* (e.g., random label probes) to assess selectivity. |

### 2.3 Diagnostic Tools Beyond Linear Probes  

* **Representational Similarity Analysis (RSA)** – compares similarity matrices of hidden states across models or layers (Kornblith *et al.*, 2019).  
* **Canonical Correlation Analysis (CCA)** – aligns subspaces of two representations to quantify shared information (Raghu *et al.*, 2017).  
* **Neuron‑level activation analysis** – identifies individual units that fire for specific concepts (e.g., “gender bias neurons” in Zhao *et al.*, 2019).  

### 2.4 Practical Tips  

1. **Use multiple probe families** (linear, MLP, decision trees) to triangulate findings.  
2. **Report control results** (random labels, shuffled inputs) to demonstrate that performance exceeds chance.  
3. **Inspect layer‑wise trends** rather than a single “best” layer; many phenomena are distributed.  

---  

## 3. Attribution and Saliency Methods for LLM Outputs  

### 3.1 From Attention to Attribution  

Early interpretability work focused on **attention weights** (e.g., Vaswani *et al.*, 2017). However, attention alone is not a faithful explanation of model decisions (Jain & Wallace, 2019). Modern attribution methods treat the model as a black‑box and compute **input‑output relevance scores**.

### 3.2 Gradient‑Based Attribution  

| Method | Core Idea | Advantages | Limitations |
|--------|-----------|------------|-------------|
| **Integrated Gradients** (Sundararajan *et al.*, 2017) | Accumulate gradients along a straight‑line path from a baseline to the actual input. | Satisfies axioms (implementation invariance, completeness). | Requires differentiable inputs; baseline choice can affect results. |
| **SmoothGrad** (Smilkov *et al.*, 2017) | Average gradients over noisy copies of the input. | Reduces gradient noise, yields smoother saliency maps. | Still sensitive to model non‑linearity. |
| **Layer‑wise Relevance Propagation (LRP)** (Bach *et al.*, 2015) | Back‑propagate relevance scores respecting conservation across layers. | Provides fine‑grained token‑level attributions. | Implementation varies across architectures. |

### 3.3 Perturbation‑Based Attribution  

* **LIME** (Ribeiro *et al.*, 2016) – locally approximates the model with an interpretable surrogate by perturbing input tokens.  
* **SHAP** (Lundberg & Lee, 2017) – estimates Shapley values for each token, guaranteeing consistency and additivity.  

These methods are model‑agnostic but can be computationally expensive for long contexts typical of LLMs. Recent adaptations (e.g., **KernelSHAP for Transformers**, Chen *et al.*, 2022) sample perturbations efficiently.

### 3.4 Visualizing Attribution in Transformers  

* **TransformerLens** (Olsson *et al.*, 2022) provides a library for extracting and visualizing attention patterns, gradients, and attribution scores across layers.  
* **Feature Attribution Maps** – heatmaps over token positions (e.g., “The **cat** sat on the **mat**”) help users see which words contributed most to a generated token.  

### 3.5 Best Practices  

1. **Combine gradient‑ and perturbation‑based methods** to cross‑validate explanations.  
2. **Report baseline selection** (e.g., zero embeddings, mean token) for Integrated Gradients.  
3. **Quantify stability** by measuring attribution variance across random seeds or input paraphrases.  

---  

## 4. Causal and Counterfactual Analysis of Model Behavior  

### 4.1 Why Causality Matters  

Attribution tells *what* contributed, but **causal analysis** asks *whether* a component truly *caused* an output. Counterfactual reasoning—asking “what would the model output if token *X* were replaced by *Y*?”—offers a stronger test of explanation fidelity.

### 4.2 Causal Mediation Techniques  

* **Causal Mediation Analysis** (Koh *et al.*, 2020) decomposes the total effect of an input feature into direct and indirect pathways through internal representations.  
* **Do‑interventions** (Pearl, 2009) are implemented by intervening on hidden states (e.g., zero‑out a neuron) and observing output changes.  

Recent work by **Mohan *et al.* (2023)** applied causal mediation to GPT‑2, revealing that *syntax‑related neurons* mediate the effect of subject–verb agreement cues on next‑token predictions.

### 4.3 Counterfactual Generation  

* **Textual Counterfactuals** – generate minimally edited inputs that flip a model’s decision (Zhang *et al.*, 2020).  
* **Counterfactual Data Augmentation** – augment training data with such edits to improve robustness and interpretability (Kumar *et al.*, 2022).  

### 4.4 Counterfactual Explanations for LLMs  

The **MICE** framework (Mothilal *et al.*, 2020) adapts counterfactual explanations to language models by optimizing a loss that balances *proximity* (few edits) and *sparsity* (few changed tokens). Applied to GPT‑3, MICE uncovered that changing a single gendered pronoun often flips downstream sentiment predictions, highlighting gender bias pathways.

### 4.5 Practical Guidelines  

1. **Isolate causal pathways** by intervening on specific layers or neuron groups rather than the entire model.  
2. **Validate counterfactuals** with human judges to ensure linguistic plausibility.  
3. **Report effect sizes** (e.g., change in log‑probability) to quantify the strength of causal influence.  

---  

## 5. Benchmarks and Metrics for Evaluating Interpretability  

Interpretability research needs **objective evaluation**. Several benchmarks and metrics have emerged to compare methods systematically.

### 5.1 Standard Benchmarks  

| Benchmark | Focus | Representative Papers |
|-----------|-------|------------------------|
| **LAMA** (Petroni *et al.*, 2019) | Factual knowledge probing via fill‑in‑the‑blank queries. | Measures whether probing reveals stored facts. |
| **GLUE‑Interpret** (Wang *et al.*, 2022) | Attribution quality on GLUE tasks (e.g., SST‑2, MNLI). | Uses human‑annotated rationales as ground truth. |
| **CausalLM** (Kumar *et al.*, 2023) | Causal mediation and counterfactual tests on language generation. | Provides a suite of “do‑intervention” tasks. |
| **ExplainBench** (Srivastava *et al.*, 2022) |
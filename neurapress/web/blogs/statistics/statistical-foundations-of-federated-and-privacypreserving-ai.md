# Statistical Foundations of Federated and Privacy‑Preserving AI  

*An in‑depth tutorial on the statistical principles that underpin modern decentralized learning systems.*

---

## 1. Introduction  

Federated learning (FL) and related privacy‑preserving paradigms have become the de‑facto approach for training high‑quality models on data that remain on‑device or on isolated silos. While the engineering challenges—communication constraints, system heterogeneity, and secure aggregation—are well documented, the **statistical underpinnings** are equally critical. Understanding how privacy mechanisms, data heterogeneity, sampling schemes, and fairness considerations interact determines whether a federated system can achieve both **utility** and **responsibility**.

This tutorial surveys the core statistical concepts that shape federated and privacy‑preserving AI:

1. **Differential privacy (DP)** guarantees and the utility‑accuracy trade‑off.  
2. **Statistical heterogeneity** and its impact on convergence.  
3. **Communication‑efficient sampling**, variance‑reduction techniques, and the resulting error bounds.  
4. **Fairness, bias, and robustness** across heterogeneous client populations.  

Each section blends rigorous results with intuitive explanations, aiming to equip researchers and practitioners with the tools needed to design provably sound FL pipelines.

---

## 2. Differential Privacy Guarantees and Utility‑Accuracy Trade‑offs  

### 2.1 Formal definition  

A randomized algorithm \(\mathcal{M}\) satisfies \((\varepsilon,\delta)\)-**differential privacy** if for any pair of adjacent datasets \(D,D'\) (differing in a single record) and any measurable set \(S\),

\[
\Pr[\mathcal{M}(D)\in S]\le e^{\varepsilon}\Pr[\mathcal{M}(D')\in S]+\delta .
\]

In federated settings, *adjacency* is typically defined at the **client level**: two federations are adjacent if they differ by the inclusion or exclusion of a single client’s entire local dataset \([Kairouz \textit{et al.}, 2021]\).

### 2.2 Gaussian mechanism for federated aggregation  

The most common DP primitive for FL is the **Gaussian mechanism** applied to the sum of client updates:

\[
\tilde{g}= \frac{1}{N}\sum_{i=1}^{N} g_i + \mathcal{N}\bigl(0,\sigma^{2}\mathbf{I}\bigr),
\]

where \(g_i\) is the (clipped) gradient contributed by client \(i\), \(N\) is the number of participating clients, and \(\sigma\) is the noise scale. **Clipping** bounds the \(\ell_{2}\)-norm of each update to a pre‑specified constant \(C\), ensuring a finite **sensitivity** \(\Delta = C/N\).

The privacy loss after \(T\) rounds can be bounded using the **moments accountant** or **RDP** (Renyi DP) composition techniques. For example, with Rényi order \(\alpha\),

\[
\varepsilon \approx \frac{T\,\alpha\,\Delta^{2}}{2\sigma^{2}} + \frac{\log(1/\delta)}{\alpha-1}.
\]

### 2.3 Utility‑accuracy trade‑off  

Adding Gaussian noise inevitably degrades the empirical risk minimization objective. A canonical trade‑off can be expressed as

\[
\underbrace{\mathbb{E}\bigl[ L(\hat{w}) - L(w^{*}) \bigr]}_{\text{excess risk}}
\;\le\;
\underbrace{\mathcal{O}\!\Bigl(\frac{C^{2}\log(1/\delta)}{N^{2}\varepsilon^{2}}\Bigr)}_{\text{privacy‑induced term}}
\;+\;
\underbrace{\mathcal{O}\!\Bigl(\frac{1}{\sqrt{NT}}\Bigr)}_{\text{statistical term}} .
\]

- **Privacy‑induced term** grows quadratically as \(\varepsilon\) shrinks (stronger privacy).  
- **Statistical term** reflects the usual convergence rate of stochastic gradient descent (SGD) under i.i.d. sampling.

Practical systems therefore tune \(\varepsilon\) in the range \([1,10]\) and adjust clipping \(C\) to balance **model accuracy** against **privacy guarantees**. Recent work on *privacy‑aware hyperparameter selection* (e.g., **Abadi \textit{et al.}, 2016**) demonstrates that modest clipping (e.g., \(C\) equal to the median gradient norm) often yields the best utility for a given \(\varepsilon\).

---

## 3. Statistical Heterogeneity and Convergence Analysis in Decentralized Training  

### 3.1 Measuring heterogeneity  

In federated learning, client datasets are typically **non‑i.i.d.**. A standard metric is the **gradient dissimilarity**:

\[
\zeta^{2} \;=\; \frac{1}{N}\sum_{i=1}^{N}\bigl\| \nabla L_i(w) - \nabla L(w) \bigr\|^{2},
\]

where \(L_i\) is the local loss on client \(i\) and \(L\) is the global loss. Larger \(\zeta\) indicates stronger heterogeneity.

### 3.2 Convergence of FedAvg under heterogeneity  

FedAvg (McMahan *et al.*, 2017) performs \(K\) local SGD steps per round before averaging. Under smoothness (\(L\)-Lipschitz gradients) and strong convexity (\(\mu\)-strongly convex), the expected distance to the optimum after \(T\) communication rounds satisfies

\[
\mathbb{E}\bigl\| w_T - w^{*} \bigr\|^{2}
\;\le\;
\underbrace{\bigl(1-\eta\mu\bigr)^{KT}\bigl\| w_0-w^{*}\bigr\|^{2}}_{\text{contraction}}
\;+\;
\underbrace{\frac{\eta L\sigma^{2}}{\mu N}}_{\text{variance term}}
\;+\;
\underbrace{\frac{\eta^{2}L^{2}\zeta^{2}}{\mu^{2}}}_{\text{heterogeneity penalty}} .
\]

- **\(\eta\)** is the local learning rate.  
- **\(\sigma^{2}\)** denotes stochastic gradient variance.  
- The **heterogeneity penalty** grows with \(\zeta^{2}\) and can dominate when data are highly skewed.

### 3.3 Algorithms that mitigate heterogeneity  

| Algorithm | Key Idea | Convergence improvement |
|-----------|----------|--------------------------|
| **FedProx** (Li *et al.*, 2020) | Adds a proximal term \(\frac{\mu}{2}\|w-w_t\|^{2}\) to each client’s objective. | Reduces the heterogeneity penalty to \(\mathcal{O}(\zeta^{2}/K)\). |
| **SCAFFOLD** (Karimireddy *et al.*, 2020) | Maintains control variates (global and local) to correct client drift. | Achieves a bound similar to i.i.d. SGD: \(\mathcal{O}\bigl(\frac{1}{\sqrt{NT}}\bigr)\). |
| **FedNova** (Wang *et al.*, 2020) | Normalizes updates by local step count, eliminating dependence on \(K\). | Guarantees linear speed‑up even with heterogeneous \(K_i\). |

These methods illustrate how **statistical modeling** of heterogeneity directly informs algorithmic design.

---

## 4. Communication‑Efficient Sampling, Variance Reduction, and Error Bounds  

### 4.1 Partial client participation  

Communicating with all \(N\) clients each round is rarely feasible. Let \(m\) be the number of **sampled** clients per round. Assuming **uniform sampling without replacement**, the unbiased estimator of the global gradient is

\[
\hat{g} = \frac{1}{m}\sum_{i\in\mathcal{S}} g_i,
\qquad \mathcal{S}\subseteq\{1,\dots,N\},\;|\mathcal{S}|=m .
\]

The variance of \(\hat{g}\) scales as \(\frac{N-m}{m(N-1)}\sigma^{2}\), motivating **importance sampling** where clients with larger gradient norms are sampled more often (Sattler *et al.*, 2019).

### 4.2 Variance‑reduction techniques  

| Technique | Principle | Resulting error bound |
|-----------|-----------|-----------------------|
| **Control variates** (e.g., SCAFFOLD) | Subtract a global drift estimate from each local update. | Reduces the stochastic term from \(\sigma^{2}\) to \(\sigma^{2}/K\). |
| **Local SGD with periodic averaging** (Stich, 2019) | Perform many local steps, average infrequently. | Achieves \(\mathcal{O}\bigl(\frac{1}{\sqrt{NT}} + \frac{L^{2}\Delta^{2}}{\mu^{2}T^{2}}\bigr)\) where \(\Delta\) is the initial distance to optimum. |
| **Gradient compression** (e.g., top‑\(k\), quantization) | Transmit a sparse/low‑precision representation. | Adds a bounded compression error \(\epsilon_{\text{comp}}\) that appears additively in the final bound. |

### 4.3 Complete error‑bound expression  

Combining **partial participation**, **variance reduction**, and **smoothness** yields a unified bound for the expected suboptimality after \(T\) rounds:

\[
\boxed{
\mathbb{E}\bigl[ L(w_T) - L(w^{*}) \bigr]
\;\le\;
\underbrace{\frac{2L\|w_0-w^{*}\|^{2}}{\mu KT}}
_{\text{optimization term}}
\;+\;
\underbrace{\frac{L\sigma^{2}}{\mu N m}}
_{\text{stochastic variance term}}
\;+\;
\underbrace{\frac{L\zeta^{2}}{\mu K}}
_{\text{heterogeneity term}}
\;+\;
\underbrace{\epsilon_{\text{comp}}}_{\text{compression error}} .
}
\]

Key observations:

- **Increasing \(K\)** (more local steps) reduces the heterogeneity term but inflates the stochastic variance unless variance‑reduction is applied.  
- **Larger \(m\)** (more sampled clients) shrinks the stochastic term linearly.  
- **Compression** introduces a controllable additive error; careful design (e.g., unbiased quantization) keeps \(\epsilon_{\text{comp}}\) negligible.

---

## 5. Fairness, Bias, and Robustness Across Heterogeneous Client Populations  

### 5.1 Fairness notions in federated learning  

1. **Group fairness** – equal performance across predefined demographic groups (e.g., parity of false‑positive rates).  
2. **Individual fairness** – similar individuals receive similar predictions, often operationalized via a Lipschitz constraint on the model w.r.t. a similarity metric.  

In FL, fairness must be evaluated **globally** (across the union of client data) while respecting **privacy** constraints that prevent direct inspection of individual records.

### 5.2 Sources of bias  

| Source | Description |
|--------|-------------|
| **Data imbalance** | Certain clients hold disproportionately many samples of a minority class. |
| **System heterogeneity** | Faster devices contribute more updates, biasing the model toward their data distribution.
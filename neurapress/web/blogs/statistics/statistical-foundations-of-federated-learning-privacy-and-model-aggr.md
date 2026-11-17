# Statistical Foundations of Federated Learning, Privacy, and Model Aggregation for Foundation Models  

*Federated learning (FL) has become the de‑facto paradigm for training massive foundation models on decentralized data. While engineering tricks make large‑scale FL feasible, the statistical guarantees that underpin safety, reliability, and efficiency are equally critical. This tutorial surveys the core statistical tools that enable practitioners to detect distribution drift and poisoning, quantify the uncertainty of an aggregated model, bound the risk introduced by differential privacy, and reason formally about the three‑way trade‑off among communication, utility, and privacy.*  

---  

## Introduction  

Foundation models such as large language models or vision transformers are typically trained on data that reside on millions of edge devices. Federated learning orchestrates **local updates** on each client and a **global aggregation** step that produces a model without ever collecting raw data. This decentralization raises three intertwined statistical challenges:

1. **Client‑distribution drift** – the data generating process on each device may evolve over time, and malicious actors may inject poisoned updates.  
2. **Uncertainty quantification** – after aggregation we need confidence intervals or Bayesian guarantees for downstream performance.  
3. **Privacy‑induced risk** – differential‑privacy (DP) mechanisms add noise, which inevitably degrades statistical risk.  

All three challenges must be balanced against the **communication budget** (number of rounds, bits per round). The sections below present a unified statistical treatment of each challenge, culminating in a principled trade‑off analysis that can guide system design for real‑world FL deployments.  

---  

## 1. Hypothesis‑Testing for Client‑Distribution Drift and Poisoning Attacks  

### 1.1 Problem Setting  

Consider a set of \(K\) clients. At round \(t\) each client \(k\) holds a local data distribution \( \mathcal{D}_{k}^{(t)}\) and computes a stochastic gradient \(\mathbf{g}_{k}^{(t)}\). The server aggregates these gradients (e.g., via FedAvg) to obtain a global update. **Distribution drift** occurs when \(\mathcal{D}_{k}^{(t)}\) deviates significantly from a reference distribution \(\mathcal{D}_{k}^{(0)}\). **Poisoning attacks** manifest as systematic bias in \(\mathbf{g}_{k}^{(t)}\) that aims to corrupt the global model.

Statistical hypothesis testing provides a principled way to flag anomalous clients:

- **Null hypothesis** \(H_{0}\): client \(k\) follows its expected distribution (no drift, no poisoning).  
- **Alternative hypothesis** \(H_{1}\): client \(k\) has deviated (drift) or is adversarial (poisoned).  

### 1.2 Two‑Sample and Kernel‑Based Tests  

A classic approach is the **maximum mean discrepancy (MMD)** test (Gretton *et al.*, 2012). Let \(\{x_{i}\}_{i=1}^{n}\) be a recent minibatch from client \(k\) and \(\{y_{j}\}_{j=1}^{m}\) a reference batch from the same client’s historic data. The unbiased MMD estimator

\[
\widehat{\text{MMD}}^{2} = \frac{1}{n(n-1)}\!\sum_{i\neq i'}\!k(x_{i},x_{i'}) + \frac{1}{m(m-1)}\!\sum_{j\neq j'}\!k(y_{j},y_{j'}) - \frac{2}{nm}\!\sum_{i,j}\!k(x_{i},y_{j})
\]

is compared against a permutation‑based null distribution to obtain a p‑value. A small p‑value indicates **distribution shift**.  

When the data are high‑dimensional, **linear‑time MMD** (Zaremba *et al.*, 2013) or **random Fourier features** (Rahimi & Recht, 2007) keep the test scalable for on‑device execution.  

### 1.3 Detecting Poisoning via Robust Statistics  

Poisoning attacks often manifest as **outlying gradient directions**. Robust hypothesis tests based on **Huber‑contaminated models** (Huber, 1964) treat each gradient as

\[
\mathbf{g}_{k}^{(t)} = (1-\epsilon)\,\mathbf{g}^{\star} + \epsilon\,\mathbf{b},
\]

where \(\epsilon\) is the contamination proportion and \(\mathbf{b}\) is an arbitrary adversarial vector. The **median‑of‑means (MoM)** estimator (Minsker, 2015) splits the set of received gradients into \(B\) blocks, computes the mean within each block, and then takes the component‑wise median. Under the null hypothesis (no attack), MoM achieves sub‑Gaussian concentration; under the alternative, the deviation exceeds a threshold that can be calibrated using a **Chernoff bound**.  

Bagdasaryan *et al.* (2020) demonstrated that a simple **norm‑threshold test** on the MoM residuals detects backdoor attacks with high probability while preserving the utility of honest updates.  

### 1.4 Practical Considerations  

| Aspect | Recommendation |
|--------|----------------|
| **Batch size** | Use at least \(n\ge 100\) samples per test to guarantee asymptotic normality of MMD. |
| **Multiple testing** | Apply a Bonferroni or Benjamini–Hochberg correction across clients to control the family‑wise error rate. |
| **Computation on device** | Prefer linear‑time MMD or sketch‑based tests to keep the overhead < 1 % of local training time. |
| **Adaptive thresholds** | Update the significance level \(\alpha\) based on the observed false‑positive rate over recent rounds. |

---  

## 2. Confidence‑Interval and Bayesian Guarantees for Aggregated Model Performance  

### 2.1 Central Limit Theorem for Federated Averaging  

FedAvg updates the global model as  

\[
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \frac{1}{K}\sum_{k=1}^{K}\mathbf{g}_{k}^{(t)} .
\]

If each client gradient is an i.i.d. draw from a distribution with mean \(\mu\) and covariance \(\Sigma\), the **multivariate CLT** yields  

\[
\sqrt{K}\,\bigl(\overline{\mathbf{g}}^{(t)} - \mu\bigr) \xrightarrow{d} \mathcal{N}\!\bigl(0,\Sigma\bigr),
\]

where \(\overline{\mathbf{g}}^{(t)}\) is the averaged gradient. This asymptotic normality enables **Wald‑type confidence intervals** for any linear functional of the model (e.g., test‑set loss).  

### 2.2 Non‑Asymptotic Concentration Bounds  

In practice, the number of participating clients per round can be modest and the data are heterogeneous. **Matrix Bernstein** (Tropp, 2012) and **empirical Bernstein** (Maurer & Pontil, 2009) inequalities give finite‑sample guarantees: for any \(\delta\in(0,1)\),

\[
\bigl\|\overline{\mathbf{g}}^{(t)} - \mu\bigr\|_{2}
\le \sqrt{\frac{2\lambda_{\max}(\Sigma)\log(2d/\delta)}{K}}
   + \frac{L\log(2d/\delta)}{3K},
\]

where \(L\) bounds the gradient norm and \(d\) is the model dimension. Plugging this bound into a **Lipschitz loss** yields a **confidence interval for the global risk**.  

Sun *et al.* (2022) derived tight **high‑probability confidence intervals** for the test accuracy of federated classifiers by combining concentration on gradients with a **Rademacher complexity** term that captures client heterogeneity.  

### 2.3 Bayesian Poster
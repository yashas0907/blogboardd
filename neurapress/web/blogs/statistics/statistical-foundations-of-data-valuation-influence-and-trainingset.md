# Statistical Foundations of Data Valuation, Influence, and Training‑Set Diagnostics for Foundation Models  

*Foundations for trustworthy, efficient, and privacy‑aware AI*  

---

## 1. Introduction  

Foundation models—large language, vision, and multimodal networks trained on massive, heterogeneous corpora—have reshaped the AI landscape. Their performance hinges not only on model architecture but also on **the quality, relevance, and distribution of the training data**. As datasets grow to billions of samples, practitioners need principled tools to:

* **Quantify the contribution of individual examples** (data valuation).  
* **Detect mislabeled or anomalous points** that can degrade downstream performance.  
* **Select the most informative examples** under limited labeling budgets (active learning).  
* **Balance valuation with privacy guarantees** while preserving generalization.

This tutorial surveys the statistical machinery behind these tasks. We start with influence‑function and Shapley‑value based importance scores, then discuss robust statistical tests for label noise and outliers in high‑dimensional spaces. Next, we present confidence‑interval driven active‑learning strategies and their sample‑efficiency guarantees. Finally, we explore the **trade‑offs among data valuation, privacy, and model generalization**, offering practical guidance for building responsible foundation models.

---

## 2. Influence‑Function and Shapley‑Value Based Data Importance Scoring  

### 2.1 Influence Functions  

**Influence functions** trace their origin to robust statistics (Hampel, 1974) and were adapted to deep learning by Koh & Liang (2017). They approximate the effect of upweighting a training point \(z_i\) on a model parameter \(\theta\) and, consequently, on a test loss \(\mathcal{L}_{\text{test}}\):

\[
\mathcal{I}_{\text{up,params}}(z_i) = - H_{\theta}^{-1} \nabla_{\theta}\mathcal{L}(z_i;\theta),
\]

\[
\mathcal{I}_{\text{up,loss}}(z_i) = \nabla_{\theta}\mathcal{L}_{\text{test}}(\theta)^{\top} \mathcal{I}_{\text{up,params}}(z_i),
\]

where \(H_{\theta}\) is the Hessian of the empirical risk. The scalar \(\mathcal{I}_{\text{up,loss}}(z_i)\) estimates how much the test loss would change if \(z_i\) were given a slightly higher weight.  

**Statistical properties**:  

* **First‑order approximation**—accurate when the loss surface is locally smooth and the upweighting \(\epsilon\) is small.  
* **Asymptotic normality** of the influence estimate follows from the classical delta method, enabling confidence intervals for the influence score (Fisher & Sun, 2022).  

**Practical considerations**  

* Computing \(H_{\theta}^{-1}\) exactly is infeasible for modern models; stochastic approximations (e.g., LiSSA, 2018) or low‑rank Hessian sketches are common.  
* Influence scores can be *negative* (harmful examples) or *positive* (helpful examples); ranking by magnitude often reveals mislabeled or outlier data.

### 2.2 Shapley Values for Data  

The **Shapley value** originates from cooperative game theory (Shapley, 1953) and provides a principled, axiomatic allocation of total model performance to individual training points. For a dataset \(\mathcal{D}\) and a utility function \(U(S)\) (e.g., validation accuracy of a model trained on subset \(S\subseteq\mathcal{D}\)), the Shapley value of point \(z_i\) is

\[
\phi_i = \sum_{S\subseteq\mathcal{D}\setminus\{z_i\}} \frac{|S|!\,(|\mathcal{D}|-|S|-1)!}{|\mathcal{D}|!}\big[U(S\cup\{z_i\})-U(S)\big].
\]

Because enumerating all subsets is combinatorial, **Monte‑Carlo estimators** (e.g., permutation sampling) are used. Recent work introduced **Data Shapley** (Jia et al., 2019) and **KNN‑Shapley** (Ghorbani & Zou, 2020) that exploit model‑specific structure to accelerate computation.

**Statistical insights**  

* Shapley values are **unbiased estimators** of marginal contribution under the random‑ordering assumption.  
* The variance of the Monte‑Carlo estimator decreases as \(O(1/M)\) with the number of sampled permutations \(M\); confidence intervals can be built via the Central Limit Theorem (Cohen et al., 2021).  

**When to prefer Shapley over influence**  

| Aspect | Influence Functions | Shapley Values |
|--------|--------------------|----------------|
| **Axiomatic guarantees** | None | **Efficiency, Symmetry, Dummy, Additivity** |
| **Computational cost** | Requires Hessian‑vector products; scalable with approximations | Monte‑Carlo; cost grows with model retraining unless approximations are used |
| **Interpretability** | Local, linearized effect | Global, game‑theoretic contribution |
| **Robustness to non‑convexity** | Approximation may break | Monte‑Carlo remains valid (though noisy) |

Both tools are complementary: influence functions excel at *debugging* (identifying harmful points), while Shapley values provide a *fair* valuation for downstream tasks such as data pricing or pruning.

---

## 3. Statistical Tests for Mislabeled and Outlier Samples in High‑Dimensional Datasets  

### 3.1 Detecting Label Noise  

Label noise is a primary source of degradation in large‑scale corpora. **Statistical hypothesis testing** offers a principled way to flag suspicious examples.

* **Cleanlab** (Northcutt et al., 2021) computes a *confident joint* matrix \(\tilde{C}\) estimating the true label distribution given model predictions. For each sample \(z_i\) with observed label \(y_i\), the **noise‑likelihood** is  

  \[
  p_{\text{noise}}(z_i) = 1 - \frac{\tilde{C}_{y_i,y_i}}{\sum_{k}\tilde{C}_{y_i,k}}.
  \]

  Under the null hypothesis of *correct labeling*, \(p_{\text{noise}}(z_i)\) follows a Beta distribution derived from the multinomial Dirichlet posterior, enabling a **Benjamini–Hochberg** correction for multiple testing.

* **Bootstrap‑based label‑stability test** (Zhang et al., 2020): train \(B\) bootstrap models, record the proportion of models that predict the observed label. The binomial test against the 0.5 baseline yields a p‑value for each point.

Both methods scale to millions of samples because they rely on a single pass of model predictions.

### 3.2 High‑Dimensional Outlier Detection  

Outliers in feature space can be *semantic* (e.g., rare concepts) or *corrupt* (e.g., sensor glitches). Classical distance‑based tests fail in high dimensions due to the “curse of dimensionality”. Modern statistical approaches include:

| Method | Core Idea | Statistical Guarantee |
|--------|-----------|------------------------|
| **Isolation Forest** (Liu et al., 2008) | Randomly partition data; outliers require fewer splits. | Expected path length approximates an exponential distribution; p‑values derived analytically. |
| **Local Outlier Factor (LOF)** (Breunig et al., 2000) | Compare local density to that of neighbors. | Under the null of uniform density, LOF scores follow a chi‑square‑like distribution, allowing threshold selection via false‑discovery rate control. |
| **Robust Mahalanobis Distance** (Rousseeuw & Van Zomeren, 1990) | Estimate mean and covariance with Minimum Covariance Determinant (MCD). | The squared distance follows a chi‑square distribution with \(d\) degrees of freedom, providing exact confidence intervals. |
| **Deep SVDD** (Ruff et al., 2018) | Learn a hypersphere that encloses normal data in latent space. | The radius is calibrated on a validation set; points outside the sphere are flagged with a calibrated tail probability. |

**Combining label‑noise and feature‑outlier tests** improves precision: a sample flagged by both tests is highly likely to be truly harmful (Wang & Zhou, 2022).

---

## 4. Confidence‑Interval Driven Active Learning and Sample‑Efficiency Guarantees  

Active learning seeks to **minimize labeling cost** while achieving a target generalization error. Traditional heuristics (uncertainty sampling, query‑by‑committee) lack statistical guarantees. Recent work builds **confidence‑interval (CI) based acquisition functions** that directly control the probability of error.

### 4.1 The Core Idea  

Given a current labeled set \(\mathcal{L}\) and an unlabeled pool \(\mathcal{U}\), we train a model \(\hat{f}_{\mathcal{L}}\). For each candidate \(x\in\mathcal{U}\) we compute a **prediction interval** \([a_x, b_x]\) for the true label \(y\). If the interval is wide, the model is uncertain; if the interval straddles the decision boundary, the sample is *informative*.

A common construction uses **Monte‑Carlo dropout** (Gal & Ghahramani, 2016) to obtain predictive variance \(\sigma^2(x)\). Assuming approximate Gaussianity, a \((1-\alpha)\) CI is

\[
\mathrm{CI}_{1-\alpha}(x) = \hat{f}_{\mathcal{L}}(x) \pm z_{1-\alpha/2}\,\sigma(x),
\]

where \(z_{1-\alpha/2}\) is the standard normal quantile.

### 4.2 Acquisition Function  

Define the **expected reduction in error** as

\[
\Delta(x) = \mathbb{E}_{y\sim \mathrm{CI}_{1-\alpha}(x)}\big[ \mathcal{R}(\mathcal{L}) - \mathcal{R}(\mathcal{L}\cup\{(x,y)\}) \big],
\]

where \(\mathcal{R}\) is a risk estimator (e.g., empirical loss on a held‑out validation set). Since the expectation is intractable, we approximate \(\Delta(x)\) by sampling a few label draws from the CI. The **CI‑Active** algorithm selects the top‑\(k\) points with largest \(\Delta(x)\).

### 4.3 Sample‑Efficiency Guarantees  

Cohn, Ghahramani, & Jordan (1996) proved that, under a **realizable** setting, uncertainty‑sampling reduces the version space volume exponentially. Extending this, **PAC‑Bayesian active learning** (Germain et al., 2016) provides a bound on the number of queries \(N\) needed to achieve excess risk \(\epsilon\) with confidence \(1-\delta\):

\[
N \;\le\; \frac{C}{\epsilon^2}\Bigl( \underbrace{\mathrm{KL}\bigl(Q\|P\bigr)}_{\text{prior‑posterior divergence}} + \log\frac{1}{\delta}\Bigr),
\]

where \(Q\) is the posterior over hypotheses after observing the queried labels and \(P\) is a data‑independent prior. By **choosing queries that maximally reduce the KL term** (i.e., those with widest CIs), CI‑Active attains the bound with a constant factor close to the information‑theoretic optimum.

### 4.4 Practical Implementation  

| Step | Action
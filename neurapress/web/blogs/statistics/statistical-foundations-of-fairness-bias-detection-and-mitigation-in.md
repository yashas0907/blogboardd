# Statistical Foundations of Fairness, Bias Detection, and Mitigation in Foundation Models  

*An in‑depth tutorial for researchers and practitioners working with large‑scale language, vision, and multimodal models.*

---

## Introduction  

Foundation models—large pretrained transformers for language, vision, and multimodal tasks—have transformed AI research, but their scale also amplifies longstanding concerns about **bias**, **unfair outcomes**, and **privacy leakage**. Addressing these issues requires a solid statistical grounding:  

1. **Disparate impact** must be quantified with reliable **hypothesis tests** and **confidence intervals**, even when model representations live in thousands of dimensions.  
2. **Sample‑selection bias** can invalidate fairness guarantees; **distribution‑free** methods are needed to protect against unknown covariate shifts.  
3. **Causal reasoning** provides a principled way to separate legitimate influence from discriminatory pathways via **mediation** and **counterfactual** decompositions.  
4. **Differential privacy (DP)** offers provable privacy, yet it interacts with utility and fairness in subtle ways that can be captured by **formal risk bounds**.

This tutorial weaves together these four pillars, offering concrete statistical tools, theoretical guarantees, and practical guidance for the next generation of responsible foundation models.

---

## 1. Statistical Tests & Confidence Intervals for Disparate Impact in High‑Dimensional Models  

### 1.1 Disparate Impact in a Nutshell  

*Disparate impact* (also called **demographic parity**) requires that a binary decision \( \hat{Y} \) be independent of a protected attribute \( A \) (e.g., gender, race):

\[
\Pr(\hat{Y}=1 \mid A=0) \approx \Pr(\hat{Y}=1 \mid A=1).
\]

The **impact ratio**  

\[
\Delta = \frac{\Pr(\hat{Y}=1 \mid A=1)}{\Pr(\hat{Y}=1 \mid A=0)}
\]

is the primary effect size. In high‑dimensional settings, \(\hat{Y}\) is often derived from a complex representation \( \mathbf{Z}\in\mathbb{R}^p\) (e.g., the final hidden layer of a transformer).  

### 1.2 Challenges of High Dimensionality  

* **Curse of dimensionality**: Classical chi‑square or Wald tests assume a fixed number of categories and a well‑behaved covariance matrix. With \(p\gg n\) (features \(\gg\) samples), the sample covariance becomes singular.  
* **Multiple testing**: Fairness audits frequently examine many subgroups (intersectional analysis), inflating Type I error.  

### 1.3 Test Statistics  

| Approach | Core Idea | When to Use |
|----------|-----------|-------------|
| **Chi‑square / Wald test** | Approximate \(\hat{\Delta}\) as a ratio of two binomial proportions; use delta method for variance. | Moderate \(p\), large \(n\). |
| **Permutation test** | Randomly permute \(A\) while keeping \(\hat{Y}\) fixed; compute empirical null distribution of \(\Delta\). | Any dimensionality; exact Type I control. |
| **Bootstrap confidence intervals** | Resample \((A,\hat{Y})\) pairs to obtain distribution of \(\Delta\). | Provides robust CIs under heteroskedasticity. |
| **High‑dimensional CLT (Chernozhukov et al., 2017)** | Uses Gaussian approximation for the supremum of empirical processes in \(p\)-dimensional space. | Very large \(p\) with sub‑Gaussian tails. |
| **Debiased Lasso inference** (Javanmard & Montanari, 2014) | Constructs asymptotically normal estimators for coefficients in sparse high‑dimensional logistic regression. | When a parametric model for \(\hat{Y}\) is fitted. |

#### Example: Wald Test for \(\Delta\)

Let \(\hat{p}_a = \frac{1}{n_a}\sum_{i:A_i=a}\hat{Y}_i\). The log‑ratio estimator  

\[
\hat{\theta}= \log \hat{\Delta}= \log \hat{p}_1 - \log \hat{p}_0
\]

has asymptotic variance  

\[
\widehat{\operatorname{Var}}(\hat{\theta}) = \frac{1-\hat{p}_1}{n_1\hat{p}_1} + \frac{1-\hat{p}_0}{n_0\hat{p}_0}.
\]

A two‑sided 95 % CI for \(\Delta\) follows from  

\[
\exp\!\Bigl(\hat{\theta} \pm 1.96\sqrt{\widehat{\operatorname{Var}}(\hat{\theta})}\Bigr).
\]

When \(p\) is large, replace the simple variance estimator with a **sandwich estimator** that accounts for dependence among the high‑dimensional features (see Belloni et al., 2014).

### 1.4 Controlling for Multiple Subgroups  

Apply the **Benjamini–Hochberg** procedure to the set of p‑values from permutation tests, or use **simultaneous confidence bands** derived from the high‑dimensional CLT to guarantee family‑wise coverage (Chernozhukov et al., 2017).

### 1.5 Practical Workflow  

1. **Extract predictions** \(\hat{Y}\) from the foundation model on a held‑out test set.  
2. **Stratify** by protected attribute(s) \(A\).  
3. **Choose** a test (e.g., permutation) based on sample size and dimensionality.  
4. **Compute** the impact ratio \(\Delta\) and its confidence interval.  
5. **Adjust** for multiple subgroups if needed.  

*Pseudo‑code* (Python‑style) is omitted for brevity but follows the standard `sklearn`/`numpy` pattern.

---

## 2. Distribution‑Free Fairness Guarantees under Sample Selection Bias  

### 2.1 What Is Sample‑Selection Bias?  

In many real‑world deployments the training data are **not** a random draw from the target population. Let \(S\in\{0,1\}\) indicate inclusion in the training set. The **selection mechanism** \(P(S=1\mid X,A,Y)\) can be unknown, leading to covariate shift or *missing‑at‑random* violations.

### 2.2 Distribution‑Free (PAC‑style) Guarantees  

A **distribution‑free** guarantee holds for any underlying data-generating distribution, relying only on finite‑sample combinatorics.

#### 2.2.1 Conformal Prediction for Fairness  

Conformal methods produce **prediction sets** \(\mathcal{C}(X)\) that contain the true label with probability \(1-\alpha\) **without** distributional assumptions (Vovk et al., 2005). By constructing *group‑conditional* conformal sets, we can guarantee:

\[
\Pr\bigl(Y\in\mathcal{C}(X) \mid A=a\bigr) \ge 1-\alpha \quad \forall a\in\{0,1\}.
\]

If the size of \(\mathcal{C}(X)\) is bounded uniformly across groups, the model satisfies a **distribution‑free equal‑coverage** fairness criterion.

#### 2.2.2 Importance‑Weighted Empirical Risk  

When a **propensity score** \(\pi_i = P(S_i=1\mid X_i,A_i)\) can be estimated (e.g., via logistic regression), the **importance‑weighted** estimator

\[
\hat{R}_{\text{IW}} = \frac{1}{n}\sum_{i=1}^n \frac{1}{\hat{\pi}_i}\,\ell\bigl(f(X_i),Y_i\bigr)
\]

is unbiased for the population risk. By applying **Hoeffding’s inequality** to the weighted sum, we obtain a **distribution‑free bound** on the disparity between groups:

\[
\bigl| \hat{R}_{\text{IW}}^{(A=0)} - \hat{R}_{\text{IW}}^{(A=1)} \bigr|
\le \sqrt{\frac{\log(2/\delta)}{2n_{\min}}}\,\frac{1}{\pi_{\min}},
\]

where \(n_{\min}\) and \(\pi_{\min}\) are the smallest group size and propensity, respectively. This bound holds **regardless** of the underlying distribution.

### 2.3 Fairness under Covariate Shift  

Lipton et al. (2018) propose **importance‑reweighted fairness constraints** that guarantee demographic parity **under covariate shift**:

\[
\bigl| \mathbb{E}_{P_{\text{test}}}[ \hat{Y}\mid A=1] - \mathbb{E}_{P_{\text{test}}}[ \hat{Y}\mid A=0] \bigr|
\le \epsilon,
\]

provided the training‑time weighted disparity satisfies the same bound. The result is **distribution‑free** because it does not assume a specific form for \(P_{\text{test}}\).

### 2.4 Summary of Guarantees  

| Guarantee | Assumption | Practical Tool |
|-----------|------------|----------------|
| Group‑conditional coverage (conformal) | Exchangeability of calibration set | `mapie` / `nonconformist` libraries |
| Weighted disparity bound | Estimated propensity scores > 0 | Logistic propensity + weighted loss |
| Covariate‑shift fairness (Lipton et al.) | Overlap: \(\pi_i>0\) for all \(i\) | Reweighting during SGD |

---

## 3. Causal Fairness Analysis via Mediation and Counterfactual Decomposition  

### 3.1 Why Causality?  

Statistical parity can be satisfied *by* ignoring legitimate predictive information. Causal analysis distinguishes **fair pathways** (e.g., education) from **unfair pathways** (e.g., direct discrimination).  

### 3.2 Causal Graphs and Notation
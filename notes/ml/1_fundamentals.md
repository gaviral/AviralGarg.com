---
layout: main
title: ML Fundamentals
---

<!--
WEBPAGE MAINTENANCE INSTRUCTIONS:
- Content must **be** quick to see and understand
- For all sections: minimize text, use Mermaid diagrams, one diagram per page unless absolutely necessary to exceed
- Diagrams should **be** simple while showing main topics
- Use horizontal space efficiently to minimize scrolling
- Each page links concepts together as in the book
- Shows progression: problem → solution → next steps
- Keep content minimal while mentioning all important details
- Comments contain exact text from the book for reference
- Diagrams should focus only on key concepts, not every detail from the text
- Newer instructions override previous ones when conflicts arise
-->

<!-- Load Mermaid from the CDN and initialize it -->
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({ startOnLoad: true });
</script>

<!-- Load MathJax for LaTeX support -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# 1. Fundamental Concepts in Machine Learning

<!-- Page 11 -->
<div class="mermaid">
graph LR
    A["**Machine Learning**"] --> B["**Classification**"]
    A --> C["**Regression**"]
    
    B --> D["**Spam Detection**<br/>e.g., Email"]
    C --> E["**Price Prediction**<br/>e.g., House"]
    
    B --> F["**Group Into**<br/>**Categories**"]
    C --> G["**Predict**<br/>**Numbers**"]
    
    F & G --> H["**Test Data**"]
    H --> I["**Compare Errors**"]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#dfd
    style E fill:#dfd
    style F fill:#dfd
    style G fill:#dfd
    style H fill:#ffe,stroke:#333
    style I fill:#ffe,stroke:#333
</div>
<!-- Page 12 -->

# 2. Cross Validation

<!-- Page 21 -->
<div class="mermaid">
graph LR
    A["**Data**"] --> B["**Cross Validation**<br/>**Splits Data**"]
    
    B --> C["**Training Set**"]
    B --> D["**Testing Set**"]
    
    subgraph Types
        E["**3-Fold CV**"]
        F["**10-Fold CV**"]
        G["**Leave-One-Out CV**"]
    end
    
    B --> E & F & G
    
    C --> H["**Find Patterns**"]
    D --> I["**Evaluate Model**"]
    
    style A fill:#ffe,stroke:#333
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
    style E fill:#dfd
    style F fill:#dfd
    style G fill:#dfd
    style H fill:#dfd
    style I fill:#dfd
</div>
<!-- Page 22 -->

# 3. Statistical Foundations

<div class="mermaid">
graph TD
    A["**Probability Distributions**"] --> B["**Types**"]
    
    B --> C["**Discrete Distributions**<br/><br/>**Key Functions**<br/>PMF: \(P(X=x)\)<br/>CDF: \(P(X\le x)\)"]
    B --> D["**Continuous Distributions**<br/><br/>**Key Functions**<br/>PDF: \(f(x)\)<br/>CDF: \(F(x)=P(X\le x)\)"]
    
    C --> E["**Binomial Distribution**<br/>$$P(x|n,p) = \frac{n!}{x!(n-x)!} p^x(1-p)^{n-x}$$<br/>Mean: \(\mu = np\)<br/>Var: \(\sigma^2 = np(1-p)\)"]
    C --> F["**Bernoulli Distribution**<br/>$$P(x|p)= p^x(1-p)^{1-x}$$<br/>Mean: \(\mu = p\)<br/>Var: \(\sigma^2 = p(1-p)\)"]
    C --> G["**Poisson Distribution**<br/>$$P(x|\lambda)= \frac{e^{-\lambda}\lambda^x}{x!}$$<br/>Mean: \(\mu = \lambda\)<br/>Var: \(\sigma^2 = \lambda\)"]
    
    D --> H["**Normal Distribution**<br/>Mean: \(\mu\), Var: \(\sigma^2\)"]
    D --> I["**Uniform Distribution**<br/>\(a\le x\le b\)<br/>Expected Frequency: $$\frac{\text{Total Frequency}}{k}$$"]
    D --> J["**Exponential Distribution**<br/>$$f(x;\lambda)= \lambda e^{-\lambda x}$$ (x ≥ 0)<br/>CDF: $$P(X\le x)= 1-e^{-\lambda x}$$"]
    D --> K["**Beta Distribution**<br/>PDF: $$f(x;\alpha,\beta)= \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$$<br/>Mean: $$E[X]=\frac{\alpha}{\alpha+\beta}$$"]
    D --> L["**T Distribution**<br/>Used for small samples<br/>with unknown population SD"]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ffe,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
    style E fill:#dfd,stroke:#333
    style F fill:#dfd,stroke:#333
    style G fill:#dfd,stroke:#333
    style H fill:#dfd,stroke:#333
    style I fill:#dfd,stroke:#333
    style J fill:#dfd,stroke:#333
    style K fill:#dfd,stroke:#333
    style L fill:#dfd,stroke:#333
</div>

<div class="mermaid">
graph TD
    A["**Bayesian Estimation**"] 
    
    A --> B["**Bayes' Theorem**<br/>$$P(\theta|data)=\frac{P(data|\theta)P(\theta)}{P(data)}$$"]
    
    A --> C["**Key Concepts**"]
    C --> D["**Likelihood Function**<br/>Measures how well<br/>parameters explain data"]
    C --> E["**Prior & Posterior**<br/>Prior updated by data<br/>to form posterior"]
    C --> F["**Sample Size Effect**<br/>Larger samples<br/>dominate prior"]
    
    A --> G["**Beta-Binomial Example**"]
    G --> H["**Prior Distribution**<br/>$$\text{Beta}(\alpha,\beta)$$"]
    G --> I["**Data Collection**<br/>x successes in n trials"]
    G --> J["**Posterior Distribution**<br/>$$\text{Beta}(\alpha+x,\beta+n-x)$$"]
    G --> K["**Posterior Mean**<br/>$$\frac{\alpha+x}{(\alpha+x)+(\beta+n-x)}$$"]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333
    style C fill:#ffe,stroke:#333
    style D fill:#dfd,stroke:#333
    style E fill:#dfd,stroke:#333
    style F fill:#dfd,stroke:#333
    style G fill:#ffe,stroke:#333
    style H fill:#dfd,stroke:#333
    style I fill:#dfd,stroke:#333
    style J fill:#dfd,stroke:#333
    style K fill:#dfd,stroke:#333
</div>

# 4. Linear Algebra & Regression

<div class="mermaid">
graph TD
    A["**Linear Algebra**"]
    
    A --> MT["**Matrix Theory**<br/><br/>
    **Rank**: Dimension of row/column space<br/>
    **Determinant**: $$a(ei-fh)-b(di-fg)+c(dh-eg)$$<br/>
    **Operations**: Addition, multiplication<br/>
    (m×n)(n×p) = (m×p)<br/>
    Non-commutative, distributive"]

    A --> VO["**Vector Operations**<br/><br/>
    **Angle**: $$\cos(\theta)= \frac{u \cdot v}{\|u\|\|v\|}$$<br/>
    **Cross Product**: $$u \times v = (u_2v_3-u_3v_2,\; u_3v_1-u_1v_3,\; u_1v_2-u_2v_1)$$"]

    A --> EA["**Eigen Analysis**<br/><br/>
    **Eigenvalues & Eigenvectors**:<br/>
    $$Av=\lambda v$$<br/>
    Sum = trace of matrix<br/>
    Product = determinant<br/><br/>
    **Orthogonal Matrices**:<br/>
    $$Q^TQ=I$$, $$\det(Q)=\pm1$$<br/>
    $$Q^{-1}=Q^T$$"]

    A --> DP["**Determinant Properties**<br/><br/>
    Row swap: multiply by -1<br/>
    Nonzero det ⇒ invertible<br/>
    $$\det(I)=1$$<br/>
    Singular if det = 0"]

    A --> R["**Regression Methods**"]

    R --> OLS["**Ordinary Least Squares**<br/><br/>
    - Minimizes squared residuals ('least squares')<br/>
    - BLUE under classical assumptions<br/>
    - Equals MLE under normal errors<br/>
    **Slope Coefficient**: change in dependent variable per unit change in independent variable<br/>
    **'Least squares'**: sum of squared differences between observed & predicted"]

    R --> MLE["**Maximum Likelihood**<br/><br/>
    Maximizes log-likelihood<br/>
    Same as OLS under normal errors"]

    R --> MSE["**Mean Squared Error (MSE)**<br/><br/>
    - MSE = SSR / n (the average of squared residuals)<br/>
    - Helps compare SSR across different sample sizes"]

    R --> M["**Models**"]
    M --> SR["**Sum of Residuals**<br/>$$\sum (y_i - \hat{y}_i)$$<br/>Issue: sign cancellation"]
    SR --> SSR2["**SSR (Sum of Squared Residuals)**<br/>$$\sum (y_i - \hat{y}_i)^2$$<br/><br/>
    - Avoids cancellation of positive/negative residuals by squaring<br/>
    - Produces a differentiable objective for gradient-based methods<br/>
    - Applies to any model shape (line, sinusoid, rocket trajectory)<br/>
    - **Cannot compare across different training data sizes** (SSR grows with more data)"]
    SSR2 --> MSE2["**MSE (Mean Squared Error)**<br/>$$\text{MSE} = \frac{\sum (y_i - \hat{y}_i)^2}{n}$$<br/><br/>
    - MSE = SSR / n (the average of squared residuals)<br/>
    - Helps compare SSR across different sample sizes"]
    MSE2 --> R2_2["**R^2 (Coefficient of Determination)**<br/><br/>
    - Proportion of variation in dependent variable explained by the model<br/>
    - Typically 0 ≤ R^2 ≤ 1<br/>
    - Dimensionless, not affected by scaling<br/>
    **SSR-based**: $$R^2 = \frac{\text{SSR(mean)} - \text{SSR(fitted)}}{\text{SSR(mean)}}$$<br/>
    **MSE-based**: $$R^2 = \frac{\text{MSE(mean)} - \text{MSE(fitted)}}{\text{MSE(mean)}}$$"]

    R --> MultiColl["**Handling Multicollinearity**<br/><br/>
    - Dropping or combining correlated variables<br/>
    - Using Ridge or Lasso regression<br/>
    - Ignoring is not recommended<br/>
    - Can lead to unstable or inflated coefficient estimates"]

    R --> Diagnostics["**Checking Model Assumptions**<br/><br/>
    - Residuals vs fitted values (patterns, homoscedasticity)<br/>
    - Residuals vs each independent variable (linearity)<br/>
    - Normal QQ plot (normality of errors)<br/>
    - 'No outliers' is *not* an official assumption<br/>
    - Hypothesis tests on coefficients do *not* check assumptions"]

    R --> Perf["**Performance Evaluation**<br/><br/>
    - Adjusted R^2 (penalizes extra predictors)<br/>
    - MSE, RMSE, etc. measure average error<br/>
    - p-values of coefficients are about significance, *not* performance<br/>
    - Plotting residuals vs X is for assumptions, *not* performance"]

    R --> GradDescent["**Gradient Descent**<br/><br/>
    - Iterative optimization method<br/>
    - Updates parameters to minimize cost function<br/>
    - Addresses 'optimization' problem"]

    A --> PCA["**Principal Component Analysis**<br/><br/>
    **Principal Components**:<br/>
    - Orthogonal (uncorrelated) axes capturing maximum variance<br/>
    - Sensitive to data scaling<br/>
    - 2D visualization: plot top 2 components<br/>
    - If first 2 PCs explain 85%, remainder is 15%"]

    R --> FeatSel["**Feature Selection**<br/><br/>
    - Reduces overfitting<br/>
    - Increases interpretability<br/>
    - Reduces computational cost<br/>
    - Not about including all features"]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style MT fill:#dfd,stroke:#333
    style VO fill:#dfd,stroke:#333
    style EA fill:#dfd,stroke:#333
    style DP fill:#dfd,stroke:#333
    style R fill:#bbf,stroke:#333
    style OLS fill:#dfd,stroke:#333
    style MLE fill:#dfd,stroke:#333
    style MSE fill:#dfd,stroke:#333
    style M fill:#dfd,stroke:#333
    style SR fill:#dfd,stroke:#333
    style SSR2 fill:#dfd,stroke:#333
    style MSE2 fill:#dfd,stroke:#333
    style R2_2 fill:#dfd,stroke:#333
    style MultiColl fill:#dfd,stroke:#333
    style Diagnostics fill:#dfd,stroke:#333
    style Perf fill:#dfd,stroke:#333
    style GradDescent fill:#dfd,stroke:#333
    style PCA fill:#dfd,stroke:#333
    style FeatSel fill:#dfd,stroke:#333

</div>

# 900. Dummy Section

<div class="mermaid">
graph TD
    A["**Dummy Root Node**<br/>_Click to go home_"]
    
    A --> B["**Category One**"]
    A --> C["**Category Two**"]
    A --> D["**Category Three**"]
    
    B --> B1["**Sub Category 1.1**"]
    B --> B2["**Sub Category 1.2**"]
    B1 --> B11["**Detail 1.1.1**"]
    B1 --> B12["**Detail 1.1.2**"]
    B2 --> B21["**Detail 1.2.1**"]
    
    C --> C1["**Sub Category 2.1**"]
    C --> C2["**Sub Category 2.2**"]
    C1 --> C11["**Detail 2.1.1**"]
    C2 --> C21["**Detail 2.2.1**"]
    C2 --> C22["**Detail 2.2.2**"]
    
    D --> D1["**Sub Category 3.1**"]
    D --> D2["**Sub Category 3.2**"]
    D1 --> D11["**Detail 3.1.1**"]
    D2 --> D21["**Detail 3.2.1**"]
    
    B11 --> E["**Shared Node**"]
    C22 --> E
    D21 --> E
    
    click A "../../index.html" "Go to Homepage"
    
    style A fill:#f9f,stroke:#0366d6,stroke-width:2px,text-decoration:underline
    style B fill:#bbf,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
    
    style B1 fill:#ffe,stroke:#333
    style B2 fill:#ffe,stroke:#333
    style C1 fill:#ffe,stroke:#333
    style C2 fill:#ffe,stroke:#333
    style D1 fill:#ffe,stroke:#333
    style D2 fill:#ffe,stroke:#333
    
    style B11 fill:#dfd,stroke:#333
    style B12 fill:#dfd,stroke:#333
    style B21 fill:#dfd,stroke:#333
    style C11 fill:#dfd,stroke:#333
    style C21 fill:#dfd,stroke:#333
    style C22 fill:#dfd,stroke:#333
    style D11 fill:#dfd,stroke:#333
    style D21 fill:#dfd,stroke:#333
    
    style E fill:#f9f,stroke:#333,stroke-width:2px
</div>

p-value explanation: https://chatgpt.com/share/67b23858-e9dc-8002-92bd-4f3edc7a2bfb
---
layout: main
title: Today I Learned
---

<!-- Load MathJax for LaTeX support -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Sat Mar 22

## 1. ML Engineering Excellence
Build portfolio demonstrating **full ML lifecycle** with emphasis on **maintenance** - highly valued by industry employers

<div class="mermaid">
flowchart LR
    PD["<b>Problem Definition</b><br><br>• Business requirement analysis<br>• Success metrics establishment"] --> 
    DE["<b>Data Engineering</b><br><br>• Collection and validation<br>• Preprocessing and cleaning<br>• Feature extraction"] --> 
    MD["<b>Model Development</b><br><br>• Algorithm selection<br>• Training and validation<br>• Hyperparameter tuning"] --> 
    DP["<b>Deployment</b><br><br>• Infrastructure setup<br>• Monitoring implementation"] --> 
    MT["<b>Maintenance</b><br><br>• Performance tracking<br>• Retraining strategies<br>• Handling data drift"]
    
    classDef highlight fill:#f9f,stroke:#333,stroke-width:2px;
    class MT highlight;
</div>

## 2. Temperature in Language Models
Controls output **randomness/determinism**
- **Setting to 0**: Maximizes predictability, ensures **consistent responses**, preferred for **production systems**

## 3. System vs User Prompts in LLMs

- **System Prompts**: Define model **behavior/personality**, establish **constraints/rules**, not visible in conversation history
- **User Prompts**: Direct queries to model, part of **visible conversation**, what end users actually input 
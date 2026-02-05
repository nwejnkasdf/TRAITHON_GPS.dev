# TRAITHON_GPS.dev
Trustworthy AI competition portfolio

## 1. Hyojung Gwon

### 1.1. Summary:
As a Technical Lead, drove product planning and built the core inference and monitoring stack for an LLM-based clickbait detection service, including 0/1 logit-based scoring, IG-based explanations, and short-/long-term drift detection with operational thresholds.

### 5.3. Detail:
- Led service planning for a platform-style product that surfaces clickbait indices aggregated by publisher, journalist, and section, with an end-to-end user + operations flow.
- Built the end-to-end service pipeline from article ingestion to scoring, storage, aggregation, and monitoring-ready outputs.
- Designed and implemented the main inference engine by constraining outputs to {0,1}, extracting the final-layer 0/1 logits, applying a 2-class softmax, and persisting p0/p1 and margin scores for downstream use.
- Developed preprocessing and XAI modules, including JSON flattening/normalization for sliceable datasets and an Integrated Gradients (IG) module for token-level attribution.
- Fine-tuned the main inference model with DoRA, improving performance from [ accuracy [0.507] / precision [0.596] / recall [0.102] / F1 [0.174] ] to [ accuracy [0.986] / precision [0.985] / recall [0.988] / F1 [0.986] ].
- Owned impact analysis end-to-end, drafting the initial impact registry, prioritizing key impacts, and assigning roles/owners across the team.
- Proposed and executed long-term drift/bias detection, designing thresholding experiments and running an operations-style simulation to translate alerts into retraining/golden-set update decisions.
- Proposed and executed short-term drift/bias detection, designing and running experiments to calibrate practical alert thresholds and response actions.
- Produced and delivered the award-ceremony presentation, translating the system into a reliability narrative (prevention–detection–response) aligned with competition evaluation criteria.

## 2. 김준호
## 3. Hyeonjeong Yoon

### 3.1. Summary:
Working as a Data Scientist, supported interpretability analysis and validation for a clickbait detection system through label integrity checks, stress testing, and experimental reporting.

### 3.2. Detail:
- Designed and refined XAI-based analysis experiments, clustering clickbait-related keywords at the human cognition level to produce interpretability-focused analysis and reporting.
- Contributed to the structuring and systematization of AI governance practices across data collection, preprocessing, modeling, and deployment, clarifying impact, risk, and response workflows.
- Conducted label integrity verification on a large-scale clickbait news dataset, quantitatively evaluating annotation reliability through inter-annotator agreement analysis.
- Designed and executed input-perturbation stress tests focused on punctuation variations, analyzing model vulnerabilities and operational risks, and documenting the results in validation reports.
- Led the creation and management of project-wide evidence artifacts, and producing presentation materials and a comprehensive project summary.

## 4. Woonjung Lee

### 4.1. Summary:
Working as a Data Scientist, conducted explainability-driven analysis and section-level bias evaluation for a clickbait detection system, and performed stress testing to identify and validate model failure modes under realistic conditions.
Also led report authoring and contributed to evidence mapping and presentation material preparation.

### 4.2. Detail:
- Designed and implemented Integrated Gradients–based XAI analysis code, and empirically validated model decision rationales through controlled experiments.
- Formulated section-level bias hypotheses based on label and subcategory distributions, and built a data proportion analysis pipeline from scratch to perform quantitative bias validation.
- Designed stress-test scenarios using disaster news datasets, modeling extreme cases where sensational lexicon is used in legitimate public-interest contexts, and verified model failure modes in which such cases are misclassified as clickbait.
- Led overall report and deliverable authoring, ensuring coherence and consistency across project outputs.
- Participated in systematic evidence mapping across experiments and analyses.
- Contributed to the development of presentation materials.

## 5. HyunJin Choi

### 5.1. Summary: 
Working as Data Scientist, performed statistical bias and robustness analysis for a clickbait detection system, including dependency tests, stress testing, and fairness monitoring design.

### 5.2. Detail:
- Defined and analyzed data bias issues in a clickbait article detection AI competition, statistically validating whether section-wise performance gaps stem from genuine section effects or structural confounding.
- Applied χ² tests and CMH conditional independence tests to disentangle section effects from processing-pattern confounders, and developed an effect-size – driven interpretation framework.
- Designed and conducted a section-swapping stress test to evaluate the impact of section–label dependency on model performance, confirming robustness to structural bias within the current data scope.
- Extended data and model analysis into an SLI/SLO-based fairness monitoring and risk management framework, proposing exposure-aware section-wise bias detection metrics and operational thresholds.

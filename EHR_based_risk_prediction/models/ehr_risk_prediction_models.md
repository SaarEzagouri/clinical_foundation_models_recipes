# Summary of Key EHR-Based Risk Prediction Models

| Model | Data Used | Training Methodology & Architecture |
| :--- | :--- | :--- |
| **BEHRT** | UK CPRD linked with HES (~1.6 million primary/secondary care patient records) | **Self-supervised pre-training + Supervised fine-tuning**: Adapted transformer architecture (BERT) trained using Masked Language Modeling (MLM) on temporal code sequences; fine-tuned for multi-label downstream disease prediction. |
| **Med-BERT** | Cerner Health Facts & Truven MarketScan (~28 million electronic health records) | **Self-supervised pre-training + Supervised fine-tuning**: Pre-trained on structural disease/procedure code histories using modified MLM with domain-specific positional embeddings; fine-tuned for readmission and heart failure risk. |
| **ClinicalBERT** | MIMIC-III database (~2.0 million unstructured clinical notes) | **Self-supervised pre-training + Supervised fine-tuning**: Pre-trained on domain-specific unstructured clinical narratives using bidirectional transformer (BERT) objectives; fine-tuned for 30-day hospital readmission risk. |
| **EHRMamba** | MIMIC-IV & eICU Collaborative Research Databases | **Self-supervised pre-training + Prompted Fine-Tuning**: Built on Selective State Space Models (Mamba); pre-trained via Next Token Prediction (NTP) on longitudinal EHR sequences and fine-tuned using Multitask Prompted Finetuning (MPF). |
| **RETAIN** | Sutter Health EHR database (~14 million visits, focused on heart failure cohorts) | **Supervised End-to-End**: Reverse-time attention mechanism built on two-level recurrent neural networks (RNNs/GRUs); trained directly on target labels (e.g., heart failure onset) to provide clinical interpretability. |
| **Deepr** | Large longitudinal hospital dataset (structured inpatient administrative/clinical records) | **Supervised End-to-End**: Convolutional Neural Network (CNN) architecture operating over mapped diagnosis and medication sequences, using explicit delimiter tokens for variable inter-visit duration. |
| **Hi-BEHRT** | CPRD & HES databases (longitudinal patient trajectories over extended timeframes) | **Self-supervised pre-training + Supervised fine-tuning**: Hierarchical Transformer architecture designed to handle extremely long histories by separating visit-level multi-head self-attention from patient-level trajectory modeling. |
| **CEHR-BERT** | Large longitudinal structured administrative & clinical datasets (OMOP CDM format) | **Self-supervised pre-training + Supervised fine-tuning**: Pre-trained using masked language modeling on standardized OMOP Common Data Model sequences, incorporating explicit temporal, visit type, and age embeddings. |
| **Doctor AI** | Sutter Health EHR database (~260,000 longitudinal patient records over multi-year spans) | **Supervised End-to-End**: Multi-layer Gated Recurrent Unit (GRU) recurrent network trained directly to predict both future disease codes and time intervals to the next visit. |
```

### Summary of Included Models
* **Transformer/SSM Models (BEHRT, Med-BERT, ClinicalBERT, EHRMamba, Hi-BEHRT, CEHR-BERT):** Utilize self-supervised pre-training (Masked Token Prediction, Next Token Prediction) followed by supervised fine-tuning for downstream clinical tasks.
* **RNN/CNN Models (RETAIN, Deepr, Doctor AI):** Utilize end-to-end supervised learning trained directly on historical sequence markers to compute specific outcomes (heart failure, readmission, next-visit risk).

Let me know if you would like to include additional models or expand on specific performance metrics for any of these systems!
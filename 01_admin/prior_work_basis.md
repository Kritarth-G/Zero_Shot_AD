# Prior Work Basis

The following papers formed the theoretical and technical foundation of this project:

### 1. WinCLIP: Zero-shot Anomaly Classification and Segmentation (Jeong et al., 2023)
* **Influence:** This is our primary base paper. We adopted the multi-scale windowing approach and the concept of Domain Adaptation Priors (DAP). The prompt templates and the sliding window feature extraction logic used in our code are directly influenced by this work.

### 2. AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models (Gu et al., 2023)
* **Influence:** We studied this paper to understand how Large Vision-Language Models (LVLMs) can be fine-tuned for anomaly localization using prompt tuning.
* **Impact on Project:** While we initially intended to implement the AnomalyGPT architecture, we found the computational requirements (high VRAM for LLM fine-tuning) to be beyond our available hardware limits. This led us to focus on optimizing the WinCLIP architecture with our own dynamic weighting module instead.
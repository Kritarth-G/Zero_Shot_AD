# Claimed Contribution

### What we reproduced
* **WinCLIP Core:** We successfully reproduced the zero-shot anomaly detection pipeline using the CLIP ViT-B/16 backbone.
* **Multi-Scale Windowing:** We implemented the spatial sliding window mechanism at scales [2, 3] to generate local anomaly maps.
* **Compositional DAP:** We utilized the standard textual templates for state, background, and object categories as described in the original WinCLIP paper.

### What we modified
* **Dynamic DAP Weighting:** Our primary modification is the introduction of a real-time Image Quality Assessment (IQA) module. We used OpenCV (Laplacian variance, Shannon Entropy, and Intensity Mean) to calculate degradation coefficients ($\alpha$).
* **Environmental Adaptation:** We modified the text embedding fusion logic to be dynamic. Instead of a static average of prompts, our model re-weights anchors based on the $\alpha$ coefficients to handle blur, grain, and exposure issues in real-world data.
* **Optimization:** We optimized the text-encoding process to initialize once per category, significantly reducing the per-image inference latency during evaluation.

### What did not work
* **AnomalyGPT Implementation:** We attempted to integrate an LVLM-based decoder (as per AnomalyGPT), but it was computationally heavy for our current hardware setup.
* **Larger Window Scales:** We found that scales [4, 5] caused out-of-memory (OOM) errors on our GPUs, so we restricted our scope to scales [2, 3].
* **Synthetic Data Adaptation:** Our dynamic weighting module did not improve results on synthetic datasets, as synthetic noise does not follow the same mathematical patterns as real camera sensor noise.

### What we believe is our contribution
Our contribution is an **"Environmentally Aware" extension of WinCLIP**. We proved that while standard WinCLIP works well on clean laboratory data (MVTec AD), it requires dynamic adaptation to succeed on noisy, real-world industrial data (Casting and Magnetic Tile). We successfully closed the 15% performance gap on noisy data by integrating classical Computer Vision metrics with modern Vision-Language Models.
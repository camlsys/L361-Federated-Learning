# L361 2025 - Final Project Proposals

This page provides guidance on choosing the Final Project for the L361 Course "Federated Learning: Theory and Practice".
Read the documentation carefully and reach out to the Teaching Assistants for further questions.

---

## Marking and Guidelines

The final project is worth **50%** of the total grade for the course.
The Final Project is due 15th March @ 12 pm; materials made public (unless you opt-out).
Deliverables for the Final Project are: Code + Report + Recorded Talk.
The talk is optional for Part II.
Separate project guidelines are provided for each cohort in this repository.
Part III/MPhils are required to be more research focused.

Choose the project during the **first two weeks of the term** (the sooner, the better).
The overarching themes/tracks suggested are proposed here.
The very details of the project have to be discussed with the teaching assistants during the lab sessions.
Groups of up to three are allowed.
Mixed (Part II with Part III/MPhil) groups are not allowed.
Students can propose projects beyond those in this document prior discussion with teaching assistants.
Multiple students/groups can pick the same theme/track.
Tracks are not provided with a recommended group size or group level (Part II or Part III/MPhil), and their suitability has to be discussed with the teaching Assistants.
**The last lab session is mostly for Project updates and discussion.**
Students can further discuss the project during the lab sessions, only.
Once a week, students can ask for booking office hours with teaching assistants if they require further support.
Teaching Assistants: Lorenzo Sani <ls985@cam.ac.uk> and Alexandru-Andrei Iacob <aai30@cam.ac.uk>

---

Here the project track follow.

## Noise Scale and Critical Batch Size in Federated Learning

- Choose your dataset (MNIST/CIFAR10/other) — choose your architecture (CNN/ViT/LM)
- Machine learning engineers have widely adopted the noise scale estimator [https://arxiv.org/abs/1812.06162] to better understand the impact of batch size for efficiently parallelizing the optimization of a wide range of neural network models. In federated learning, some have tried to draw a comparison between large batches and large cohorts of clients (the number of client samples per round) [https://arxiv.org/abs/2106.07820]. However, it is unclear how the noise scale and the critical batch size frameworks can translate to federated learning. This track suggests looking into a simple experimental framework that could be theoretically or heuristically justified for bridging the gap between centralized and federated under the lens of the noise scale and critical batch size framework.
- Any recent development or research direction along the lines of this project track that meets the curiosity of students can be discussed with the teaching Assistants.

---

## Adaptive Optimizers with Local Updates

- Choose your dataset (MNIST/CIFAR10/other) — choose your architecture (CNN/ViT/LM)
- Recently, optimizers with local updates (theoretical basics for federated optimization) have become increasingly popular. In particular, the widespread adoption of AdamW [https://arxiv.org/abs/1711.05101] for optimizing LLM pre-training has brought researchers and practitioners to experiment with practical means of using adaptive optimizers with local updates [https://arxiv.org/abs/2409.13155]. Recent theoretical work has shed light on the convergence guarantees of some of these variants. This track suggests looking into validating these findings with experiments and, potentially, stretching their assumptions to assess their validity and practicality.
- Alternative approaches that can be similarly investigated are:
  - Aggregating only momenta of local adaptive optimizers while maintaining the model parameter personalized locally.
  - Implementing variants of the DeMO [https://arxiv.org/abs/2411.19870] method, which, despite the lack of theoretical and technical depth, sounds particularly interesting.
- Any other recent development or research direction along the lines of this project track that meets the curiosity of students and teaching Assistants can be discussed with the teaching Assistants.

---

## Student-Proposed System Theme

- Federated learning has a strong system component that strongly interacts with the machine learning pipeline that we want to perform. There are several challenges under the federated learning system umbrella that are worth investigating. Any practical or theoretical aspect under this umbrella, such as reproducing a research paper or implementing a subcomponent of a federated learning system, can be proposed as a project prior discussion with the teaching Assistants. The student is encouraged to have a read for the more deployment-orientated paper discussing federated learning in a practical setting. However, a small reminder for the student is to be aware of the resources available to you from the lab - having an industry-scale FL experiment is exciting, but it may be hard to find a resource to run for.

---

## Configurable Aggregation under Dynamic Topology

- The aggregation step during a federated round is critical to effectively learn from the local training performed at clients. In some cases, this represents a substantial overhead for the time efficiency of the federated learning into the wild. Assuming that model parameters exchange in a federated environment doesn’t represent a privacy concern, the orchestrator would rather take advantage of HPC techniques to perform the aggregation operation in a distributed manner without using the parameter server paradigm. We assume to adopt a federated optimization based on the aggregation of pseudo gradients (the distance between the global model at the start of the round and the local models partially trained by the clients at the end of the round). There exist a series of simple algorithms adopted in HPC, such as AllReduce, Ring-AllReduce, Hierarchical Aggregation or a mixture of the above. However, in a federated learning environment, the topology between clients (the network bandwidth between them) is highly variable and heterogeneous. To allow for developing methods for dynamically choosing the optimal distributed, the communication stack for the model parameters in a federated learning framework would require to be configurable given some directive. PyTorch Distributed can be used to do so. The objective of this track is to implement and benchmark a PyTorch Distributed-based communication stack that can execute from the simple aggregation algorithms, such as Ring-AllReduce, to more complex mixed algorithms given a set of directives. The track can be pushed further to more complex scenarios and learnable solutions.

---

## They Train Large but Communicate Little

- Federated learning settings always try to take the most out of the distributed data of the clients while lowering at best the overhead of local training and communication. When local computing resources are significant, but the interconnects between the clients and the server provide very low bandwidth, it is critical to take advantage of the computing by reducing the impact of the communication. In such a setting, research has proposed means of communicating partial updates or freezing layers. In most cases, the simple low-hanging fruit solution reveals to be the best. This track aims at reproducing published research results or propose a preliminary investigation of an advanced or low-hanging fruit solution for communicating little while training large models locally. Recent research paper example is [https://arxiv.org/abs/2410.11559v3]

---

## Being Imprecise Alone but Precise Together

- [Low Precision Local Training is Enough for Federated Learning](https://neurips.cc/virtual/2024/poster/93183) present @ NeurIPS’24
- Federated learning usually runs on edge devices (e.g., mobile phones, smartwatches, etc.), which constrain the computational power and would have limitations on the throughput and effectiveness of federated learning. However, one of the recent publications at NeurIPS’24 has shown that you could do a ‘delegation of power’ back to the aggregation server by only training in lower precision (i.e., 8-bit or 6-bit quantization) on local hardware and do the full FP16/FP32 quantization on the server to reduce the computation requirement on the local device and still remain surprisingly good on the performance. It would be further interesting to explore the usefulness of such a method under a more extensive setting, for example:
  - Discover whether the so-called “quantization collapse” effect exists for such an approach on a specific client scale and quantization scale; if so, are they agnostic to the task/dataset/heterogeneity? (choose your dataset + architecture)
  - Extend the setting to other practical scopes (choose your task: ASR, NLP, etc.) Given the vastness of LLM quantization techniques available, it may be particularly interesting to investigate whether this would allow more practical LLM pre-training/fine-tuning under a low precision quantize training setting; if so, which techniques would enable a more scalable training setup under various client settings.
  - Implement a hardware-accelerated version of such an algorithm on the selected platform of your choice (e.g., Nvidia Jetson, Qualcomm dev board, Google TPU) and measure the real-world impact of such techniques. Deduce whether the numerical format provided by the target platform has a positive impact on training throughput and how realistic it is to deploy such techniques in the wild.

---

## Cycle-Consistent Multi-Model Merging (C2M3)

The goal of this project is to implement the Cycle-Consistent Multi-Model Merging (C2M3) algorithm in the Flower framework, conduct experiments on model merging and federated learning (FL), and analyse performance under various conditions using a range of models and datasets.

It follows a list of potential project tasks and experiments to be conducted within this project theme.

### C2M3 1. Implementation of C2M3 in Flower

- Understand the C2M3 algorithm from the provided paper. The algorithm aims to merge neural networks by optimizing for permutation symmetries in weight space, ensuring cycle consistency.
- Develop the C2M3 aggregation algorithm as a module in the Flower framework.
- Implement functionality for model merging that maps neural networks into a universal parameter space using the Frank-Wolfe algorithm.

### C2M3 2. Experiments on Model Merging

#### C2M3 2.1: Experiment Setup

- Use models with architectures such as ResNet18, small CNNs, small language models, and next-character prediction models (e.g., LSTMs).
- Datasets include FEMNIST, CIFAR-100, The Pile, and Shakespeare.

#### C2M3 2.2: Experiments

1. **Model Initialization Consistency:**
   - Train models on IID data for an increasing number of epochs/steps and merge them at the end.
   - Repeat the above with non-IID data distributions.
   - Analyse how the performance of the merged model is influenced by training duration and data heterogeneity.
   - Compare with standard methods like model averaging and the TIES algorithm (https://arxiv.org/pdf/2306.01708)
2. **Varying Initializations:**
   - Train models starting with different random initializations.
   - Merge models trained on IID and non-IID data distributions.
   - Compare performance with identical initialization experiments.

### C2M3 3. Federated Learning with C2M3

#### C2M3 3.1: Experiment Setup

- Configure the Flower framework for FL simulations with clients using the selected models and datasets.

#### C2M3 3.2: Experiments

1. **Identical Initialization:**
   - All clients begin with the same model initialization.
   - Train on IID data with varying aggregation frequencies, and local learning rates.
   - Repeat with non-IID data distributions.
   - Compare with standard methods like FedAvg, FedOPT e.t.c
2. **Diverse Initializations:**
   - Use different model initializations for each client.
   - Test on both IID and non-IID data distributions.
  
### C2M3 4. Performance Analysis

- Evaluate model accuracy, loss, and convergence for each experiment.
- Analyse the effect of dataset heterogeneity and initialization diversity on merging quality.
- Assess computational costs and runtime for the implemented algorithm compared to baselines.
- **TIES-Merging algorithm**

---

## AutoFLIP

### Objective

This project aims to reproduce the key findings from the AutoFLIP paper, which introduces a novel loss-based adaptive hybrid pruning method for Federated Learning (FL). By replicating its experiments and conducting additional evaluations, the project will analyse AutoFLIP’s performance across diverse FL scenarios. The reproduction will focus on verifying its ability to improve model accuracy, reduce computational and communication overheads, and enhance convergence in non-IID environments.

It follows a list of potential project tasks and experiments to be conducted within this project theme.

### AutoFLIP 1. Reproduction of AutoFLIP

- Implement the AutoFLIP method as described in the paper, focusing on:
  - The **loss exploration phase** to compute parameter relevance.
  - Adaptive pruning based on the **pruning guidance matrix**.
- Integrate the method into the Flower FL framework for experimental evaluation.

### AutoFLIP 2. Experimentation and Evaluation

- Reproduce the experiments outlined in the paper using the same datasets, models, and hyperparameters. Key aspects to verify:
  - Performance improvements in **non-IID settings**.
  - Compression ratio and communication cost reductions.
  - Convergence behavior compared to baseline FL algorithms.
- Use small CNNs, small language models, or small next-character prediction models
- Datasets: FEMNIST, CIFAR-100, The Pile, Shakespeare

### AutoFLIP 3. Extend the Evaluation

- Explore additional settings and configurations not detailed in the paper, such as:
  - Varying pruning thresholds and exploration epochs.
  - Alternative client selection strategies and aggregation frequencies.
  - Additional comparisons with pruning baselines like L1, L2, and Random Pruning.

### AutoFLIP 4. Comparative Analysis

- Evaluate AutoFLIP’s performance against baseline FL methods, such as **FedAvg**, **FedProx**, and **PruneFL**. Metrics for comparison include:
  - Accuracy, compression ratio, and communication efficiency.
  - Computational cost at both client and server levels.
  - Robustness in extreme non-IID conditions.

---

## Ties-Merging

The project focuses on implementing the **TIES-Merging algorithm** in the Flower framework, studying its effectiveness in model merging and federated learning (FL) scenarios. This includes:

1. Addressing parameter interference caused by redundant parameter changes and sign conflicts.
2. Testing under IID and non-IID settings with varying model initializations and hyperparameters.
3. Comparing the performance of TIES-Merging against FL baselines like **FedAvg**, **FedOPT**, and **FedProx**.

It follows a list of potential project tasks and experiments to be conducted within this project theme.

### Ties-Merging 1. Implementation of TIES-Merging in Flower Framework

- **Algorithm Implementation:**
  - Implement the **Trim, Elect Sign & Merge (TIES-Merging)** algorithm based on its three stages:
    1. **Trim**: Identify and zero out redundant parameters by keeping only the top-k(%) largest task vector values.
    2. **Elect**: Resolve sign conflicts across models by aggregating sign vectors and selecting the most common sign per parameter.
    3. **Disjoint Merge**: Average only parameters with the agreed-upon sign.
  - Include a scaling parameter `λ` to adjust the magnitude of the merged task vector.
- **Hyperparameter Support**:
  - Allow flexible configurations for:
    - Trimming percentage (`k`).
    - Scaling factor (`λ`).
    - Learning rate and training steps during downstream task evaluation.

### Ties-Merging 2. Experiments: Model Merging with TIES-Merging

#### Ties-Merging 2.1: Experiment Setup

- **Models**: ResNet18, small CNNs, small language models, and next-character prediction models (e.g., LSTMs).
- **Datasets**: FEMNIST, CIFAR-100, The Pile, and Shakespeare.

#### Ties-Merging 2.2: Experiments

1. **Effect of Redundant Parameter Trimming:**
   - Vary the trimming percentage `k` (e.g., 20%, 30%, 50%) and analyze the impact on the merged model’s performance.
2. **IID vs. Non-IID Data Distributions:**
   - Train models on IID data and merge them after varying epochs.
   - Repeat the same on non-IID data and compare the merged model’s performance.
3. **Model Initialization:**
   - Test the algorithm with both identical and diverse model initializations.
   - Analyze how initialization impacts merging quality and performance.
4. **Parameter Sensitivity:**
   - Study the effect of varying hyperparameters (`λ`, `k`) on merging outcomes.

### Ties-Merging 3. Federated Learning with TIES-Merging

#### Ties-Merging 3.1: Experiment Setup

**FL Configuration**:

- Use Flower to simulate FL scenarios with multiple clients.
- Experiment with different aggregation frequencies (e.g., every 1, 5, or 10 epochs).

#### Ties-Merging 3.2: Experiments

1. **Performance under IID and Non-IID Distributions:**
   - Test TIES-Merging on FL tasks with IID and non-IID client datasets.
   - Vary the number of clients and data partitions to test scalability.
2. **Impact of Aggregation Frequency:**
   - Compare frequent (e.g., per epoch) vs. sparse aggregation (e.g., every 10 epochs).
   - Analyze trade-offs between computation and performance.
3. **Comparison with FL Baselines:**
   - Compare TIES-Merging against FedAvg, FedOPT, and FedProx in terms of:
     - Model accuracy/loss.
     - Convergence speed.

### Ties-Merging 4. Comparative Analysis

**TIES-Merging vs. Baseline Merging Algorithms:**

Evaluate TIES-Merging’s effectiveness against methods like Task Arithmetic, Fisher Merging, and RegMean for both model merging and FL.
  
**FL Baselines vs. TIES-Merging:**

Highlight scenarios where TIES-Merging surpasses or falls short of FedAvg, FedOPT, and FedProx.

---

## Empirical Epsilon

The objective of this project is to reproduce the experiments and results presented in the paper, **"One-Shot Empirical Privacy Estimation for Federated Learning"** (ICLR 2024). The focus will be on implementing the proposed one-shot empirical privacy estimation framework, validating its theoretical findings, and empirically evaluating its efficacy under varying conditions.

It follows a list of potential project tasks and experiments to be conducted within this project theme.

### Empirical Epsilon 1. Understanding and Setting Up the Framework

**Algorithm Implementation:**

- Implement the **one-shot privacy estimation algorithm** (Algorithm 1) for the Gaussian mechanism.
- Extend this to the federated learning setup (Algorithm 2) to estimate privacy loss under DP-FedAvg.
- For advanced scenarios, implement Algorithm 3, which includes privacy estimation using all intermediate model updates.

**Dependencies:**

- Identify and install all dependencies for the federated learning setup, including frameworks like TensorFlow Federated or PySyft, and libraries for differential privacy (e.g., Opacus or TensorFlow Privacy).

### Empirical Epsilon 2. Reproducing Experiments

#### Experiment 2.1: One-Shot Privacy Estimation for the Gaussian Mechanism

Use synthetic datasets to test Algorithm 1:

- Generate datasets `D` and neighboring datasets `D'` with isotropically distributed canaries.
- Compare estimated privacy parameters (`ε`) for different noise levels, dimensionalities (`d`), and numbers of canaries (`k`).

Validate the convergence of the cosine-based privacy estimator to analytical bounds of the Gaussian mechanism.

#### Experiment 2.2: Privacy Estimation in DP-FedAvg

Reproduce experiments with federated learning setups:

- Dataset: Use StackOverflow word prediction as the primary dataset and EMNIST for comparison.
- Model: Word-based LSTM with ~4M parameters for StackOverflow, and CNN for EMNIST.
- Training: Implement DP-FedAvg with adaptive clipping and Gaussian noise addition.
- Privacy Estimation: Insert canary clients with isotropic updates and measure cosine similarities between their updates and the final model.

Compare privacy estimates (`ε`) under different noise levels and adversarial threat models:

- Final-model-only adversary.
- Full-iterate adversary.

#### Experiment 2.3: Effect of Multiple Canary Presentations

Investigate the impact of client participation frequency on privacy leakage:

- Simulate scenarios where canaries participate 1, 2, 4, or 8 times.
- Measure separability of observed and unobserved canaries using cosine distributions.
- Evaluate how privacy metrics (e.g., ε) change with increasing canary presentations.

#### Experiment 2.4: Hyperparameter Sensitivity Analysis

Vary parameters such as:

- Noise multiplier.
- Clipping norm.
- Learning rate (both client and server).
- Aggregation frequency.

Analyse the impact on privacy leakage (`ε`) and model utility (accuracy).

---

## [TBD] Unlearning Theme from Meghdad

---

## [TBD] FL on Smartphones Theme from DQ

---

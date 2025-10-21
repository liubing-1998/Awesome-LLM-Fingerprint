# 🔥Awesome-LLM-Fingerprint
🚀An UP-TO-DATE collection list for Large Language Model (LLM) Fingerprinting

## 📖Table of Contents

- [🔥Awesome-LLM-Fingerprint](#awesome-llm-fingerprint)
  - [📖Table of Contents](#table-of-contents)
    - [1. 🔍大模型指纹技术（LLM Fingerprinting）](#1-大模型指纹技术llm-fingerprinting)
      - [1.1 🔤参数指纹（Parameter Fingerprints）：](#11-参数指纹parameter-fingerprints)
      - [1.2 📝表征指纹（Representation Fingerprints）：](#12-表征指纹representation-fingerprints)
      - [1.3 🧠输出指纹（Output Fingerprints）：](#13-输出指纹output-fingerprints)
    - [2.🕵️大模型水印技术（LLM Watermark）](#2️大模型水印技术llm-watermark)
      - [2.1 🛡️预训练阶段水印（Pre-training Watermark）：](#21-️预训练阶段水印pre-training-watermark)
      - [2.2 🔍微调阶段水印（Fine-tuning Watermark）](#22-微调阶段水印fine-tuning-watermark)
      - [2.3 🎯推理阶段水印（Inference Watermark）](#23-推理阶段水印inference-watermark)
    - [3 🎭other](#3-other)

### 1. 🔍大模型指纹技术（LLM Fingerprinting）

大模型指纹利用大模型的固有特征来构建模型身份，具有环境不变性、特异性与不可伪造性。

我们依据指纹所依赖的特征来源，对大模型指纹方法进行分类

#### 1.1 🔤参数指纹（Parameter Fingerprints）：

基于对模型内部结构与参数的直接刻画，捕捉架构与参数的特异模式，构建可识别的模型身份。

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| Fingerprint Vector： Enabling Scalable and Efficient Model Fingerprint Transfer via Vector Addition |                     |        |        |
| HuRef：HUman-Readable Fingerprint for Large Language Models  |                     |        |        |
| Stealing Part of a Production Language Model                 |                     |        |        |
| UTF：Undertrained Tokens as Fingerprints A Novel Approach to LLM Identification |                     |        |       
|               |        | |

#### 1.2 📝表征指纹（Representation Fingerprints）：

是对大模型的间接刻画方式，通过向模型输入大量样本，分析其在表征空间中的隐表征和梯度特征，从中提取具备特异性的特征模式，用于构建模型身份。

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| Emergent Response Planning in LLMs                           |                     |        |        |
| EverTracer：Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint |                     |        |        |
| FDLLM：A Text Fingerprint Detection Method for LLMs in Multi-Language, Multi-Domain Black-Box Environments |                     |        |        |
| FDLLMA ：Dedicated Detector for Black-Box LLMs Fingerprinting |                     |        |        |
| Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification |                     |        |        |
| Leveraging Fuzzy Fingerprints from Large Language Models for Authorship Attribution |  |  | |
| MEraser：An Effective Fingerprint Erasure Approach for Large Language Models | | | |
| REEF representation encoding fingerprints for large language models | | | |
| Riemannian-Geometric Fingerprints of Generative Models | | | |
| | | |



#### 1.3 🧠输出指纹（Output Fingerprints）：

是对大模型的间接刻画，通过分析大模型的多模态输出或 logits 输出，提取具有特异性的输出行为特征，以构建模型身份。

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| A Fingerprint for Large Language Models |                     |        |        |
| CoTSRF：Utilize Chain of Thought as Stealthy and Robust Fingerprint of Large Language Models |                     |        |        |
| Detecting Stylistic Fingerprints of Large Language Models |                     |        |        |
| DuFFin：A Dual-Level Fingerprinting Framework for LLMs IP Protection |                     |        |        |
| EditMF：Drawing an Invisible Fingerprint for Your Large Language Models |                     |        |        |
| From Text to Source：Results in Detecting Large Language Model-Generated Content |                     |        |        |
| Hide and seek：fingerprinting large language models with evolution learning |                     |        |        |
| Idiosyncrasies in Large Language Models |                     |        |        |
| I'm Spartacus, No, I'm Spartacus：Measuring and Understanding LLM Identity Confusion |                     |        |        |
| Invisible Traces：Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps |                     |        |        |
| LLMmap：Fingerprinting for Large Language Models |                     |        |        |
| LLMs Have Rhythm：Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis |                     |        |        |
| Natural Fingerprints of Large Language Models |                     |        |        |
| ProFLingo：A Fingerprinting-based Intellectual Property Protection Scheme for Large Language Models |                     |        |        |
| RAP-SM：Robust Adversarial Prompt via Shadow Models for Copyright Verification of Large Language Models |                     |        |        |
| RouteMark：A Fingerprint for Intellectual Property Attribution in Routing-based Model Merging |                     |        |        |
| StegGuard_Secrets_Encoder_and_Decoder_Act_as_Fingerprint_of_Self-Supervised_Pretrained_Model |                     |        |        |
| TRAP：Targeted Random Adversarial Prompt Honeypot for Black-Box Identification |                     |        |        |
| Your Large Language Models Are Leaving Fingerprints |                     |        |        |
|              |                     |        |        |
|              |                     |        |        |

### 2.🕵️大模型水印技术（LLM Watermark）

大模型水印是一种主动身份注入机制。通过微调训练等方式，向模型注入特异性的标识信息，以实现可控的模型身份识别。

我们根据水印的注入阶段，对大模型水印方法进行分类

#### 2.1 🛡️预训练阶段水印（Pre-training Watermark）：

在模型预训练中，通过对训练目标或参数优化的控制，形成具有独特模式的参数结构。

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes |                     |        |        |
|              |                     |        |        |


#### 2.2 🔍微调阶段水印（Fine-tuning Watermark）

在微调过程中注入特定的输入–输出映射（trigger–response pairs），使模型在接收到特定触发样本时产生可识别的响应。

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| EditMark：Training-free and Harmless Watermark for Large Language Models |                     |        |        |
| Filtering_Resistant_Large_Language_Model_Watermarking_via_Style_Injection |                     |        |        |
| FPEdit：Robust LLM Fingerprinting through Localized Knowledge Editing |                     |        |        |
| FP-VEC： Fingerprinting Large Language Models via Efficient Vector Addition |                     |        |        |
| GAI-AntiCopy_Infrequent_Transformation_Aided_Accuracy-Consistent_Copyright_Protection_for_Generative_AI_Instructions_in_NGN |                     |        |        |
| Have you merged my model ？on the robustness of large language model ip protection methods against model merging |                     |        |        |
| Hey, That’s My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique |                     |        |        |
| ImF：Implicit Fingerprint for Large Language Models |                     |        |        |
| Improved Unbiased Watermark for Large Language Models |                     |        |        |
| Instructional Fingerprinting of Large Language Models |                     |        |        |
| Large Language Models as Carriers of Hidden Messages |                     |        |        |
| Learnable Fingerprints for Large Language Models |                     |        |        |
| LLM Fingerprinting via Semantically Conditioned Watermarks |                     |        |        |
| Mark your llm：Detecting the misuse of open-source large language models via watermarking |                     |        |        |
| mergeprint：Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models |                     |        |        |
| Robust LLM Fingerprinting via Domain-Specific Watermarks |                     |        |        |
| Scalable Fingerprinting of Large Language Models -ICLR |                     |        |        |
| TIBW： Task-Independent Backdoor Watermarking with Fine-Tuning Resilience for Pre-Trained Language Models |                     |        |  |
| Turning Your Strength into Watermark：Watermarking Large Language Model via Knowledge Injection | | | |
| Watermarking Makes Language Models Radioactive | | | |

#### 2.3 🎯推理阶段水印（Inference Watermark）

通过在推理阶段控制采样策略或对 logits进行轻量扰动，在不访问和改变模型参数的条件下，在输出文本中形成可识别的身份特征。

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| A Semantic Invariant Robust Watermark for Large Language Models |                     |        |        |
| An Unforgeable Publicly Verifiable Watermark for Large Language Models |                     |        |        |
| Black-Box Detection of Language Model Watermarks |                     |        |        |
| Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation？ |                     |        |        |
| Can Watermarked LLMs be Identified by Users via Crafted Prompts？ |                     |        |        |
| Discovering Spoofing Attempts on Language Model Watermarks |                     |        |        |
| Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks |                     |        |        |
| Espew：Robust copyright protection for llm-based eaas via embedding-specific watermark |                     |        |        |
| Mark your llm：Detecting the misuse of open-source large language models via watermarking |                     |        |        |
| Revisiting the Robustness of Watermarking to Paraphrasing Attacks |                     |        |        |
| Robust and Minimally Invasive Watermarking for EaaS |                     |        |        |
| Ward：Provable RAG Dataset Inference via LLM Watermarks |                     |        |        |
| Watermarking language models through language models |                     |        |        |
| Watermarking_LLMs__Challenges_and_Opportunities_in_Electronic_Design_Automation |                     |        |        |
| WaterSeeker：Pioneering Efficient Detection of Watermarked Segments in Large Documents |                     |        |        |
|              |                     |        |        |

### 3 🎭other


| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| Copyright Protection for Large Language Models：A Survey of Methods, Challenges, and Trends |                     |        |        |
| Position We Need An Algorithmic Understanding of Generative AI |                     |        |        |
| SoK：Large Language Model Copyright Auditing via Fingerprinting |                     |        |        |
| Watermark Stealing in Large Language Models |                     |        |        |
| Watermarking for Large Language Models：A Survey |                     |        |        |
| Watermarking Large Language Models and the Generated Content：Opportunities and Challenges |                     |        |        |
|              |                     |        |        |


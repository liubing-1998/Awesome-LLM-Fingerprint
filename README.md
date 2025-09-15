# 🔥Awesome-LLM-Fingerprint
🚀An UP-TO-DATE collection list for Large Language Model (LLM) Fingerprinting

## 📖Table of Contents

- [🔥Awesome-LLM-Fingerprint](#awesome-llm-fingerprint)
  - [📖Table of Contents](#table-of-contents)
    - [1. 🔍Fingerprinting LLMs: Vectors of Analysis](#1-fingerprinting-llms-vectors-of-analysis)
      - [1.1 🔤Distributional Vector:](#11-distributional-vector)
      - [1.2 📝Lexical Vector:](#12-lexical-vector)
      - [1.3 🧠Syntactic Vector:](#13-syntactic-vector)
      - [1.4 Semantic (Embedding) Vector:](#14-semantic-embedding-vector)
    - [2. 🕵️Watermarking LLMs: Vectors of Injection](#2-️watermarking-llms-vectors-of-injection)
      - [2.1 🛡️Token Selection Vector (Decoding-Time Injection):](#21-️token-selection-vector-decoding-time-injection)
      - [2.2 🔍Syntactic Vector (Structural Injection):](#22-syntactic-vector-structural-injection)
      - [2.3 🎯Semantic Vector (Meaning-Space Injection):](#23-semantic-vector-meaning-space-injection)
      - [2.4 📝Post-Hoc Vector (Output-Text Injection):](#24-post-hoc-vector-output-text-injection)
    - [3 📊 Related Survey](#3--related-survey)
    - [4 🎭other](#4-other)

### 1. 🔍Fingerprinting LLMs: Vectors of Analysis

The Core Idea: Statistical Traces in Generated Text: 
Explain how LLMs, despite their creativity, leave subtle statistical patterns in their output. 
This section introduces the concept of analyzing these patterns through different vectors.


#### 1.1 🔤Distributional Vector:

Focus: Analyzing the core probability distribution of the text.

Methods: Calculating perplexity or log-likelihood of text under a candidate model. Analyzing model logits/softmax outputs (when available). This is the most direct fingerprint.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅Huref: Human-readable fingerprint for large language models | 2024/NeurIPS | HuRef introduces human-readable fingerprints for LLMs using stable parameter vector directions and invariant terms robust to rearrangements, mapped to Gaussian vectors and images with StyleGAN2, verified via ZKP. | [arXiv](https://arxiv.org/abs/2312.04828), [Code](https://github.com/LUMIA-Group/HuRef)      |
| ✅Natural fingerprints of large language models | 2025 | Explores natural fingerprints in LLMs, revealing outputs distinguishable via unigram and Transformer classifiers, even on identical data, due to training variations like random seeds and hyperparameters. | [arXiv](https://arxiv.org/abs/2504.14871) |
| ✅Stealing part of a production language model | 2024/ICML  | This paper presents a model-stealing attack extracting the embedding projection layer from black-box production LLMs via API queries, recovering hidden dimensions and matrices with low cost. | [arXiv](https://arxiv.org/abs/2403.06634), [code](https://github.com/dpaleka/stealing-part-lm-supplementary)       |
| ✅A fingerprint for large language models | 2024 | This paper proposes a black-box fingerprinting for LLMs using output logits' vector spaces to verify ownership via space similarity, robust to PEFT attacks, without training or fine-tuning. | [arXiv](https://arxiv.org/abs/2407.01235), [code](https://github.com/solitude-alive/llm-fingerprint)       |



#### 1.2 📝Lexical Vector:

Focus: Analyzing word choice and vocabulary.

Methods: N-gram frequencies, vocabulary richness (Type-Token Ratio), usage of specific or rare words.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅Natural fingerprints of large language models | 2025 | Explores natural fingerprints in LLMs, revealing outputs distinguishable via unigram and Transformer classifiers, even on identical data, due to training variations like random seeds and hyperparameters. | [arXiv](https://arxiv.org/abs/2504.14871) |
| ✅ UTF: Undertrained tokens as fingerprints a novel approach to llm identification | 2025 | Introduces UTF, a novel LLM fingerprinting method using under-trained tokens for efficient, black-box identification with minimal performance impact, robust to fine-tuning. | [arXiv](https://arxiv.org/pdf/2410.12318) |
| ✅ Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique | 2025  | Introduces Chain & Hash, a black-box LLM fingerprinting method using hashed question-answer pairs, ensuring transparency, efficiency, and robustness against transformations with minimal utility impact. | [arXiv](https://arxiv.org/pdf/2407.10887)|
|  |  |  | |

#### 1.3 🧠Syntactic Vector:

Focus: Analyzing sentence structure and grammar.

Methods: Distribution of sentence lengths, parse tree complexity, frequency of different Part-of-Speech (POS) tag sequences.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
|  |           |  | [arXiv](#), [Code](#) |

#### 1.4 Semantic (Embedding) Vector:

Focus: Analyzing the meaning and high-dimensional representation of text.

Methods: Analyzing the distribution of text embeddings in a vector space (e.g., using centroids, variance, or specialized classifiers on embeddings).

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅ Reef: Representation encoding fingerprints for large language models |  ICLR 2025 Oral | Summary: Proposes REEF, a training-free fingerprinting method using CKA similarity on LLM representations to identify model derivations, robust to fine-tuning, pruning, merging, and permutations. | [arXiv](https://arxiv.org/abs/2410.14273),[Code](https://github.com/AI45Lab/REEF) |
| ✅ From text to source: Results in detecting large language model-generated content | 2024           | This study explores cross-model detection and attribution of LLM-generated text, revealing an inverse relationship between classifier effectiveness and model size, with promising results in watermarking detection but no quantization signatures. | [arXiv](https://arxiv.org/abs/2309.13322)       |
| ✅ Idiosyncrasies in large language models | ICML 2025 poster     | Studies LLM idiosyncrasies through text classification, achieving 97.1% accuracy on five models; identifies word distributions and semantic patterns as key, robust to rewriting and summarization. | [arXiv](https://arxiv.org/abs/2502.12150), [code](https://github.com/locuslab/llm-idiosyncrasies)      |
| ✅❓ Fdllm: A text fingerprint detection method for llms in multi-language multi-domain black-box environments | 2025 | FDLLM introduces a dedicated black-box LLM fingerprinting method using LoRA fine-tuning, achieving a 22.1% higher Macro F1 score than baselines, with robust performance against adversarial attacks on a 90,000-sample bilingual dataset. | [arXiv](https://arxiv.org/abs/2501.16029v1)       |
| ✅ Natural fingerprints of large language models |  2025  | Explores natural fingerprints in LLMs, revealing outputs distinguishable via unigram and Transformer classifiers, even on identical data, due to training variations like random seeds and hyperparameters. | [arXiv](https://arxiv.org/pdf/2504.14871v1) |
| ✅ Hide and seek: Fingerprinting large language models with evolutionary learning |  2024  | Introduces "Hide and Seek," a black-box method using an Auditor and Detective LLM to fingerprint model families with 72% accuracy, leveraging semantic manifolds and evolutionary learning. | [arXiv](https://arxiv.org/pdf/2408.02871), [Code](https://github.com/MorpheusAIs/HideNSeek) |
| ✅ MERGEPRINT: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models | ACL 2025 | MERGEPRINT introduces a merge-resistant fingerprinting method for black-box ownership verification of LLMs, embedding robust fingerprints that survive model merging, optimized against pseudo-merged models with minimal performance impact. | [arXiv](https://arxiv.org/pdf/2410.08604)|
| ✅ LLMmap: Fingerprinting for Large Language Models | USENIX 2025 | Introduces LLMmap, an active fingerprinting technique identifying 42 LLM versions with 95% accuracy using 3-8 crafted queries, robust across RAG and diverse configurations. | [arXiv](https://arxiv.org/pdf/2407.15847), [Code](https://github.com/pasquini-dario/LLMmap) |
| ✅ Have You Merged My Model? On The Robustness of Large Language Model IP Protection Methods Against Model Merging | LAMPS ’24 | This study evaluates the robustness of LLM IP protection methods, like Quantization Watermarking and Instructional Fingerprint, against model merging techniques, finding fingerprints more resilient than watermarks. | [arXiv](https://arxiv.org/pdf/2404.05188), [Code](https://github.com/ThuCCSLab/MergeGuard)|
|  |           |  | |



### 2. 🕵️Watermarking LLMs: Vectors of Injection

The Core Idea: Modifying the Generation Process:
Explain how watermarking involves subtly influencing the LLM's output by injecting a signal at a specific point in the generation pipeline.


#### 2.1 🛡️Token Selection Vector (Decoding-Time Injection):

Focus: Manipulating the choice of the next token during autoregressive generation.

Methods: Using a secret key to partition the vocabulary into "green" and "red" lists and favoring the green list tokens. Modifying the logits (the raw scores for each token) before the softmax function is applied.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅ ImF: Implicit Fingerprint for Large Language Models |           | Proposes ImF, an implicit fingerprint method embedding semantically correlated QA pairs in LLMs, resisting GRI attacks and ensuring robust ownership verification under adversarial conditions. | [arXiv](https://arxiv.org/pdf/2503.21805) |
| ✅ Large Language Models as Carriers of Hidden Messages | 2025 | This paper shows that fine-tuning embeds hidden text in LLMs, triggered by specific queries for fingerprinting or steganography, but introduces an extraction attack (UTF) and a defense (UTFC) to enhance security. | [arXiv](https://arxiv.org/pdf/2406.02481), [Code](https://github.com/kubaaa2111/zurek-stegano)|
|  |           |  |  |

#### 2.2 🔍Syntactic Vector (Structural Injection):

Focus: Embedding a signal in the grammatical structure of the output.

Methods: Guiding the model to produce specific, uncommon but valid, syntactic patterns that encode the watermark. This is a less common but powerful theoretical vector.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
|  |           |  | [arXiv](#), [Code](#) |

#### 2.3 🎯Semantic Vector (Meaning-Space Injection):

Focus: Embedding a signal in the semantic content or meaning.

Methods: Steering the model towards using certain synonyms or phrasing that subtly encodes information. This is highly complex and largely theoretical.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅ Instructional fingerprinting of large language models |  2024 naacl-long  | This paper introduces InstructionalFingerprint, a lightweight method using instruction backdoors with confidential keys to fingerprint LLMs, ensuring ownership verification post-finetuning without impacting normal performance. | [arXiv](https://arxiv.org/pdf/2401.12255), [Code](https://cnut1648.github.io/Model-Fingerprint/) |
| ✅ UTF: Undertrained tokens as fingerprints a novel approach to llm identification | 2025 | Introduces UTF, a novel LLM fingerprinting method using under-trained tokens for efficient, black-box identification with minimal performance impact, robust to fine-tuning. | [arXiv](https://arxiv.org/pdf/2410.12318), [code](https://github.com/imjccai/fingerprint) |
| ✅ Scalable fingerprinting of large language models |  ICLR 2025 long paper | This paper proposes Perinucleus sampling, a scalable fingerprinting method adding 24,576 fingerprints to a Llama-3.1-8B model without degrading utility, persisting post-finetuning and mitigating security risks. | [arXiv](https://openreview.net/pdf?id=ImrmzMDq5z) |
|  |           |  |  |

#### 2.4 📝Post-Hoc Vector (Output-Text Injection):

Focus: Modifying the text after it has been fully generated.

Methods: Synonym replacements, imperceptible character additions (e.g., zero-width spaces), or slight rephrasing. These methods do not alter the generation process itself.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
|  |           |  | [arXiv](#), [Code](#) |


### 3 📊 Related Survey

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
|  | 2024 |  | [arXiv](https://arxiv.org/pdf/2312.02003), [Code](#) |

### 4 🎭other


| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ⛔️ Detecting Stylistic Fingerprints of Large Language Models | 2025 |  | [arXiv](https://arxiv.org/abs/2503.01659) |
| ⛔️ Fingerprint Vector: Enabling Scalable and Efficient Model Fingerprint Transfer via Vector Addition | 2025 |  | [arXiv](https://arxiv.org/pdf/2409.08846), [Code](https://github.com/Xuzhenhua55/Fingerprint-Vector) |
| ⛔️ I’m Spartacus, No, I’m Spartacus:  Measuring and Understanding LLM Identity Confusion | 2024 |  | [arXiv](https://arxiv.org/pdf/2411.10683) |
| ⛔️ Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps | 2025 |  | [arXiv](#), [Code](#) |
| ⛔️ Learnable Fingerprints for Large Language Models | 学生毕业论文 |  | [arXiv](#), [Code](#) |
| ⛔️ Leveraging Fuzzy Fingerprints from Large Language Models for Authorship Attribution | 2024 |  | [arXiv](https://ieeexplore.ieee.org/document/10612177) |
| ⛔️ LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis | 2025 |  | [arXiv](https://arxiv.org/pdf/2502.20589v1) |
| ⛔️ ProFLingo: A Fingerprinting-based Intellectual Property Protection Scheme for Large Language Models | 2024 |  | [arXiv](https://arxiv.org/pdf/2405.02466), [Code](https://github.com/hengvt/ProFLingo) |
| ⛔️ TRAP: Targeted Random Adversarial Prompt Honeypot for Black-Box Identification | ACL2024 |  | [arXiv](https://arxiv.org/pdf/2402.12991), [Code](https://github.com/parameterlab/trap) |
| ⛔️ Your Large Language Models Are Leaving Fingerprints | 2024 |  | [arXiv](https://arxiv.org/pdf/2405.14057) |
| ⛔️ EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint | 2025 |  | [arXiv](https://arxiv.org/pdf/2509.03058), [Code](https://github.com/Xuzhenhua55/EverTracer) |
|  | |  |  |
|  | |  |  |
|  | |  |  |
|  | |  |  |
| ❓Mark your llm: Detecting the misuse of open-source large language models via watermarking | ICLR 2025 | | [arXiv](https://arxiv.org/abs/2503.04636) |
| ❓CoTSRF: Utilize Chain of Thought as Stealthy and Robust Fingerprint of Large Language Models | 2025 | | [arXiv](https://arxiv.org/abs/2505.16785) |
| ❓DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection | 2025 | | [arXiv](https://arxiv.org/abs/2505.16530), [Code](https://github.com/yuliangyan0807/llm-fingerprint) |
| ❓EditMark: Training-free and Harmless Watermark for Large Language Models | 2025 | | [pdf](https://openreview.net/pdf?id=qGLzeD9GCX) |
| ❓EditMF: Drawing an Invisible Fingerprint for Your Large Language Models | 2025 | | [arXiv](https://arxiv.org/abs/2508.08836) |
| ❓Espew: Robust copyright protection for llm-based eaas via embedding-specific watermark | 2025 | | [pdf](https://openreview.net/pdf?id=BltNzMweBY) |
| ❓Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification | 2025 | | [arXiv](https://arxiv.org/abs/2506.01631) |
| ❓MEraser: An Effective Fingerprint Erasure Approach for Large Language Models | ACL 2025 | | [arXiv](https://arxiv.org/abs/2506.12551), [Code](https://github.com/JingxuanZhang77/MEraser) |
| ❓RAP-SM: Robust Adversarial Prompt via Shadow Models for Copyright Verification of Large Language Models | 2025 | | [arXiv](https://arxiv.org/abs/2505.06304v1) |
| ❓Riemannian-Geometric Fingerprints of Generative Models | 2025 | | [arXiv](https://arxiv.org/abs/2506.22802) |
| ❓Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes | PETs 2025 | | [pdf](https://www.petsymposium.org/popets/2025/popets-2025-0126.pdf) |
| ❓Robust and Minimally Invasive Watermarking for EaaS | ACL 2025 | | [arXiv](https://arxiv.org/abs/2410.17552) |
| ❓Robust LLM Fingerprinting via Domain-Specific Watermarks | 2025 | | [arXiv](https://arxiv.org/abs/2505.16723), [Code](https://github.com/eth-sri/robust-llm-fingerprints) |
| ❓RouteMark: A Fingerprint for Intellectual Property Attribution in Routing-based Model Merging | 2025 | | [arXiv](https://arxiv.org/abs/2508.01784) |
| ❓TIBW: Task-Independent Backdoor Watermarking with Fine-Tuning Resilience for Pre-Trained Language Models | MDPI 2025 | | [mdpi](https://www.mdpi.com/2227-7390/13/2/272) |
| ❓Turning Your Strength into Watermark: Watermarking Large Language Model via Knowledge Injection | 2024 | | [arXiv](https://arxiv.org/abs/2311.09535) |
| ❓Watermarking language models through language models | 2025 | | [arXiv](https://arxiv.org/abs/2411.05091) |
| ❓Copyright Protection for Large Language Models: A Survey of Methods, Challenges, and Trends | 2025 | | [arXiv](https://arxiv.org/abs/2508.11548), [Code](https://github.com/Xuzhenhua55/awesome-llm-copyright-protection) |
| ❓SoK: Large Language Model Copyright Auditing via Fingerprinting | 2025 | | [arXiv](https://arxiv.org/abs/2508.19843), [Code](https://github.com/shaoshuo-ss/LeaFBench) |
| ❓Watermarking Large Language Models and the Generated Content: Opportunities and Challenges | IEEE 2025 | | [arXiv](https://arxiv.org/abs/2410.19096) |
| ❓StegGuard: Secrets Encoder and Decoder Act as Fingerprint of Self-Supervised Pre-Trained Model | IEEE 2025 | | [ieee](https://ieeexplore.ieee.org/document/11071981) |
| ❓GAI-AntiCopy: Infrequent Transformation Aided Accuracy-Consistent Copyright Protection for Generative AI Instructions in NGN | IEEE 2025 | | [ieee](https://ieeexplore.ieee.org/document/10838598) |
| ❓Watermarking LLMs—Challenges and Opportunities in Electronic Design Automation | IEEE 2025 | | [ieee](https://ieeexplore.ieee.org/document/11125763) |
| ❓Filtering Resistant Large Language Model Watermarking via Style Injection | IEEE 2025 | | [ieee](https://ieeexplore.ieee.org/document/10888201) |
| ❓Fragile Fingerprinting to Validate Training Data for Large Language Models | 2025(学位论文) | | [proQuest](https://www.proquest.com/openview/3702070489345c3f8fc0463484e4a87f/1?pq-origsite=gscholar&cbl=18750&diss=y) |
|  |           |  |  |



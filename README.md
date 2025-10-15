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
      - [2.5 📝Parameter Vector Injection:](#25-parameter-vector-injection)
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
| ✅ EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint | 2025 | EverTracer introduces a stealthy, robust gray-box fingerprinting framework for LLMs, embedding ownership via natural data memorization and verifying through probability variations. | [arXiv](https://arxiv.org/pdf/2509.03058), [Code](https://github.com/Xuzhenhua55/EverTracer) |
| ✅ ProFLingo: A Fingerprinting-based Intellectual Property Protection Scheme for Large Language Models | 2024 | ProFLingo introduces a black-box fingerprinting scheme for LLMs, generating optimized queries to elicit target responses from originals and verify derivatives via high target response rates. | [arXiv](https://arxiv.org/pdf/2405.02466), [Code](https://github.com/hengvt/ProFLingo) |
| ⛔️ I’m Spartacus, No, I’m Spartacus:  Measuring and Understanding LLM Identity Confusion | 2024 | The paper investigates identity confusion in LLMs, finding 25.93% of 27 models affected, often due to hallucinations, impacting trust and requiring improved design transparency. | [arXiv](https://arxiv.org/pdf/2411.10683) |
| ❓RouteMark: A Fingerprint for Intellectual Property Attribution in Routing-based Model Merging | 2025 | RouteMark introduces a finetune-free fingerprinting framework for IP attribution in MoE model merging, leveraging routing logits to create task-discriminative expert fingerprints robust against tampering. | [arXiv](https://arxiv.org/abs/2508.01784) |
| ❓❓Black-Box Detection of Language Model Watermarks   | ICLR 2025 | The paper introduces a pioneering statistical approach to detect LLM watermarks, focusing on Red-Green, Fixed-Sampling, and Cache-Augmented schemes, and reveals their detectability in black-box settings using real-world models like GPT-4. | [arXiv](https://arxiv.org/abs/2405.20777),[Code](https://github.com/eth-sri/watermark-detection) |
| ❓❓Discovering Spoofing Attempts on Language Model Watermarks   | ICML 2025 | Gloaguen et al. (2025) introduce a statistical method to detect spoofing in LLM watermarks by analyzing token color sequence dependence on context, offering a promising defense against learning-based spoofing attacks. | [arXiv](https://arxiv.org/abs/2410.02693) |
| ❓❓Watermarking Makes Language Models Radioactive | NeurIPS 2024 | Sander et al. (2024) explore how watermarking large language models (LLMs) induces "radioactivity," enabling detection of synthetic text usage in training, with high confidence (p-value < 10⁻⁵) even when only 5% of data is watermarked, as demonstrated through statistical tests and real-world fine-tuning experiments.  | [arXiv](https://arxiv.org/abs/2402.14904),[Code](https://github.com/facebookresearch/radioactive-watermark) |
|  |  |  | |

#### 1.2 📝Lexical Vector:

Focus: Analyzing word choice and vocabulary.

Methods: N-gram frequencies, vocabulary richness (Type-Token Ratio), usage of specific or rare words.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅Natural fingerprints of large language models | 2025 | Explores natural fingerprints in LLMs, revealing outputs distinguishable via unigram and Transformer classifiers, even on identical data, due to training variations like random seeds and hyperparameters. | [arXiv](https://arxiv.org/abs/2504.14871) |
| ✅ UTF: Undertrained tokens as fingerprints a novel approach to llm identification | 2025 | Introduces UTF, a novel LLM fingerprinting method using under-trained tokens for efficient, black-box identification with minimal performance impact, robust to fine-tuning. | [arXiv](https://arxiv.org/pdf/2410.12318) |
| ✅ Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique | 2025  | Introduces Chain & Hash, a black-box LLM fingerprinting method using hashed question-answer pairs, ensuring transparency, efficiency, and robustness against transformations with minimal utility impact. | [arXiv](https://arxiv.org/pdf/2407.10887)|
| ✅ TRAP: Targeted Random Adversarial Prompt Honeypot for Black-Box Identification | ACL2024 | TRAP uses optimized adversarial suffixes to force a target LLM to output predefined responses to random prompts, enabling black-box identity verification with >95% true positive rate. | [arXiv](https://arxiv.org/pdf/2402.12991), [Code](https://github.com/parameterlab/trap) |
| ⛔️ Your Large Language Models Are Leaving Fingerprints | 2024 | Large language models leave unique linguistic fingerprints in generated text through subtle n-gram and POS frequency differences, enabling simple classifiers for robust detection and model attribution across domains. | [arXiv](https://arxiv.org/pdf/2405.14057) |
|  |  |  | |

#### 1.3 🧠Syntactic Vector:

Focus: Analyzing sentence structure and grammar.

Methods: Distribution of sentence lengths, parse tree complexity, frequency of different Part-of-Speech (POS) tag sequences.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅ Detecting Stylistic Fingerprints of Large Language Models | 2025 | An ensemble of three classifiers with unanimous voting detects stylistic fingerprints in texts from Claude, Gemini, Llama, and OpenAI LLMs, achieving 99.88% precision and revealing similarities in unseen models. | [arXiv](https://arxiv.org/abs/2503.01659) |
| ⛔️ Your Large Language Models Are Leaving Fingerprints | 2024 | Large language models leave unique linguistic fingerprints in generated text through subtle n-gram and POS frequency differences, enabling simple classifiers for robust detection and model attribution across domains. | [arXiv](https://arxiv.org/pdf/2405.14057) |
|  |           |  | |

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
| ⛔️ Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps | 2025 | This paper introduces a hybrid fingerprinting framework combining static probing and dynamic output classification to identify underlying LLMs in real-world GenAI apps, achieving up to 86.5% accuracy despite multi-agent and access challenges. | [arXiv](#), [Code](#) |
| ⛔️ Leveraging Fuzzy Fingerprints from Large Language Models for Authorship Attribution | 2024 | This paper leverages fuzzy fingerprints from fine-tuned large language models like RoBERTa to create unique author signatures via hidden unit activations, achieving state-of-the-art authorship attribution on IMDb62 and Blog datasets with reduced model size. | [arXiv](https://ieeexplore.ieee.org/document/10612177) |
| ❓CoTSRF: Utilize Chain of Thought as Stealthy and Robust Fingerprint of Large Language Models | 2025 | CoTSRF extracts LLM fingerprints from Chain-of-Thought reasoning patterns using contrastive learning on augmented responses, enabling stealthy black-box infringement verification via KL divergence, outperforming prior methods against perturbations. | [arXiv](https://arxiv.org/abs/2505.16785) |
| ❓DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection | 2025 | DuFFin is a black-box dual-level fingerprinting framework for LLM IP protection, extracting trigger-pattern embeddings and knowledge answers to verify ownership of fine-tuned or quantized pirated models with >0.95 IP-ROC accuracy. | [arXiv](https://arxiv.org/abs/2505.16530), [Code](https://github.com/yuliangyan0807/llm-fingerprint) |
| ❓RAP-SM: Robust Adversarial Prompt via Shadow Models for Copyright Verification of Large Language Models | 2025 | RAP-SM proposes a framework using shadow models for joint optimization of adversarial suffixes, extracting shared fingerprints from LLM series to enable robust copyright verification for homologous downstream models. | [arXiv](https://arxiv.org/abs/2505.06304v1) |
| ❓Riemannian-Geometric Fingerprints of Generative Models | 2025 | Song and Itti propose a Riemannian-geometric framework for generative model fingerprints, defining artifacts as deviations from real data manifolds learned via VAE pullback metrics, enabling superior attribution across 27 architectures and modalities. | [arXiv](https://arxiv.org/abs/2506.22802) |
| ❓StegGuard: Secrets Encoder and Decoder Act as Fingerprint of Self-Supervised Pre-Trained Model | IEEE 2025 | StegGuard introduces a fingerprinting method for self-supervised pretrained models, using a learned secrets encoder-decoder pair to embed/extract bits from embeddings, achieving robust piracy detection with low error using few queries. | [ieee](https://ieeexplore.ieee.org/document/11071981) |
| ❓❓FPEdit: Robust LLM Fingerprinting through Localized Knowledge Editing | 2025 |FPEdit introduces a robust LLM fingerprinting method using localized knowledge editing, embedding natural language fingerprints that resist fine-tuning and detection, ensuring reliable ownership verification with minimal performance impact. | [arXiv](https://arxiv.org/abs/2508.02092) |
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
| ❓Robust LLM Fingerprinting via Domain-Specific Watermarks | 2025 | This paper proposes domain-specific watermarking for robust LLM fingerprinting, embedding KGW signals only in targeted subdomains like French or math to achieve reliable detection, finetuning persistence, and preserved quality. | [arXiv](https://arxiv.org/abs/2505.16723), [Code](https://github.com/eth-sri/robust-llm-fingerprints) |
| ❓❓An Unforgeable Publicly Verifiable Watermark for Large Language Models | ICLR 2024 | Liu et al. (2024) propose the UPV algorithm, an unforgeable publicly verifiable watermark for large language models, using separate neural networks for generation and detection, achieving near 99% F1 scores while resisting forgery, as detailed in their ICLR 2024 paper. | [arXiv](https://arxiv.org/abs/2307.16230),[Code](https://github.com/THU-BPM/unforgeable_watermark) |
| ❓❓Improved Unbiased Watermark for Large Language Models   | 2025 | The paper introduces MCMARK, an advanced unbiased watermarking technique that enhances detectability by over 10% and maintains text quality by partitioning vocabulary into segments, as validated through experiments on LLAMA-3. | [arXiv](https://arxiv.org/abs/2502.11268),[Code](https://github.com/RayRuiboChen/MCMark) |
| ❓❓Revisiting the Robustness of Watermarking to Paraphrasing Attacks | EMNLP 2024 | The study revisits the robustness of watermarking techniques for large language models, revealing that reverse-engineering green lists with 200K tokens enables paraphrasing attacks that reduce detection rates below 10%, challenging the security of schemes like UNIGRAM-WATERMARK and SIR. | [arXiv](https://arxiv.org/abs/2411.05277),[Code](https://github.com/codeboy5/revisiting-watermark-robustness) |
| ❓❓Ward: Provable RAG Dataset Inference via LLM Watermarks      | ICLR 2025 | formalizes RAG Dataset Inference (RAG-DI), introduces the FARAD dataset, and proposes WARD, a watermark-based method offering statistical guarantees for detecting unauthorized data usage in RAG systems, outperforming baselines. | [arXiv](https://arxiv.org/abs/2410.03537),[Code](https://github.com/eth-sri/ward) |
| ❓❓Watermark Stealing in Large Language Models | ICML 2024 | The paper reveals that querying a watermarked LLM API allows attackers to reverse-engineer watermarking rules, enabling spoofing and scrubbing attacks with over 80% success for under $50, challenging the robustness of current schemes. | [arXiv](https://arxiv.org/abs/2402.19361) |
| ❓❓WaterSeeker: Pioneering Efficient Detection of Watermarked Segments in Large Documents | NAACL 2025 Findings; AAAI PDLM Workshop (Oral) | WaterSeeker, developed by Pan et al. (2025), pioneers efficient detection of watermarked segments in large documents, using a "first locate, then detect" strategy to balance accuracy and time complexity, outperforming traditional full-text methods. | [arXiv](https://arxiv.org/abs/2409.05112),[Code](https://github.com/THU-BPM/WaterSeeker) |
|  |           |  |  |

#### 2.2 🔍Syntactic Vector (Structural Injection):

Focus: Embedding a signal in the grammatical structure of the output.

Methods: Guiding the model to produce specific, uncommon but valid, syntactic patterns that encode the watermark. This is a less common but powerful theoretical vector.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ❓GAI-AntiCopy: Infrequent Transformation Aided Accuracy-Consistent Copyright Protection for Generative AI Instructions in NGN | IEEE 2025 | GAI-AntiCopy proposes infrequent syntactic transformations like emphasis and passivization to embed watermarks into generative AI instructions, enabling accuracy-consistent copyright protection in next-generation networks without semantic distortion. | [ieee](https://ieeexplore.ieee.org/document/10838598) |
| ❓❓Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation? | ACL 2025 | The paper investigates the vulnerability of LLM watermarking, proposing pre- and post-distillation removal techniques like targeted paraphrasing and watermark neutralization, revealing limitations in preventing unauthorized knowledge transfer. | [arXiv](https://arxiv.org/abs/2502.11598) |
| ❓❓Can Watermarked LLMs be Identified by Users via Crafted Prompts? | ICML 2024 | The paper explores how text watermarking for LLMs, while effective in detecting outputs and preventing misuse, can be identified by users through crafted prompts, introducing the Water-Probe method to detect watermark biases. | [arXiv](https://arxiv.org/abs/2410.03168),[Code](https://github.com/THU-BPM/Watermarked_LLM_Identification) |
| ❓❓Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks | NeurIPS 2025 poster  |The paper "Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks" introduces the SEEK scheme, breaking the trade-off between scrubbing and spoofing robustness by using equivalent texture keys, achieving significant gains of +24.6% and +92.3% respectively.| [arXiv](https://arxiv.org/abs/2507.06274) |
|  |           |  | |

#### 2.3 🎯Semantic Vector (Meaning-Space Injection):

Focus: Embedding a signal in the semantic content or meaning.

Methods: Steering the model towards using certain synonyms or phrasing that subtly encodes information. This is highly complex and largely theoretical.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅ Instructional fingerprinting of large language models |  2024 naacl-long  | This paper introduces InstructionalFingerprint, a lightweight method using instruction backdoors with confidential keys to fingerprint LLMs, ensuring ownership verification post-finetuning without impacting normal performance. | [arXiv](https://arxiv.org/pdf/2401.12255), [Code](https://cnut1648.github.io/Model-Fingerprint/) |
| ✅ UTF: Undertrained tokens as fingerprints a novel approach to llm identification | 2025 | Introduces UTF, a novel LLM fingerprinting method using under-trained tokens for efficient, black-box identification with minimal performance impact, robust to fine-tuning. | [arXiv](https://arxiv.org/pdf/2410.12318), [code](https://github.com/imjccai/fingerprint) |
| ✅ Scalable fingerprinting of large language models |  ICLR 2025 long paper | This paper proposes Perinucleus sampling, a scalable fingerprinting method adding 24,576 fingerprints to a Llama-3.1-8B model without degrading utility, persisting post-finetuning and mitigating security risks. | [arXiv](https://openreview.net/pdf?id=ImrmzMDq5z) |
| ❓EditMark: Training-free and Harmless Watermark for Large Language Models | 2025 | EditMark introduces a training-free, harmless watermarking method for LLMs, leveraging model editing and answer diversity to embed 8-bit watermarks efficiently with near-100% extraction success. | [pdf](https://openreview.net/pdf?id=qGLzeD9GCX) |
| ❓EditMF: Drawing an Invisible Fingerprint for Your Large Language Models | 2025 | EditMF proposes a training-free LLM fingerprinting method that embeds imperceptible ownership signals via model editing of encrypted fictional author-novel-protagonist triples, achieving high robustness and negligible performance loss. | [arXiv](https://arxiv.org/abs/2508.08836) |
| ❓Espew: Robust copyright protection for llm-based eaas via embedding-specific watermark | 2025 | ESpeW introduces embedding-specific watermarking for LLM-based EaaS, injecting unique signals at distinct low-magnitude positions to resist removal attacks while preserving embedding quality. | [pdf](https://openreview.net/pdf?id=BltNzMweBY) |
| ❓Robust and Minimally Invasive Watermarking for EaaS | ACL 2025 | This paper proposes ESpeW, a robust, minimally invasive watermarking for EaaS that injects unique position-specific signals into embeddings, resisting removal attacks while altering less than 1% of quality. | [arXiv](https://arxiv.org/abs/2410.17552) |
| ❓TIBW: Task-Independent Backdoor Watermarking with Fine-Tuning Resilience for Pre-Trained Language Models | MDPI 2025 | TIBW embeds task-independent backdoor watermarks into pre-trained language models using semantically dissimilar trigger-target pairs and parameter relationship embedding, ensuring fine-tuning resilience and high performance across NLP tasks. | [mdpi](https://www.mdpi.com/2227-7390/13/2/272) |
| ❓Turning Your Strength into Watermark: Watermarking Large Language Model via Knowledge Injection | 2024 | This paper proposes a novel LLM watermarking method via knowledge injection, embedding encoded watermarks into customizable knowledge (e.g., code functions) and fine-tuning the model for black-box extraction with near-100% success, ensuring fidelity, stealth, and robustness. | [arXiv](https://arxiv.org/abs/2311.09535) |
| ❓Watermarking language models through language models | 2025 | A prompt-guided watermarking framework uses three cooperating LMs to dynamically embed imperceptible signals in outputs via user-specific instructions, enabling black-box detection and robustness across diverse models. | [arXiv](https://arxiv.org/abs/2411.05091) |
| ❓Filtering Resistant Large Language Model Watermarking via Style Injection | IEEE 2025 |  it embeds watermarks by fine-tuning LLMs on semantically preserved but stylistically altered inputs , steering the model to produce specific outputs for those inputs without altering core meaning, while enhancing resistance to filtering attacks through imperceptible style triggers. | [ieee](https://ieeexplore.ieee.org/document/10888201) |
| ❓❓A Semantic-Invariant Robust Watermark for Large Language Models | ICLR 2024 |  propose a semantic invariant robust watermark for large language models, achieving high accuracy in detecting LLM-generated text by embedding watermarks based on semantic embeddings, ensuring robustness against attacks like paraphrasing and enhancing security.  | [arXiv](https://arxiv.org/abs/2310.06356),[Code](https://github.com/THU-BPM/Robust_Watermark) |
|  |           |  |  |

#### 2.4 📝Post-Hoc Vector (Output-Text Injection):

Focus: Modifying the text after it has been fully generated.

Methods: Synonym replacements, imperceptible character additions (e.g., zero-width spaces), or slight rephrasing. These methods do not alter the generation process itself.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ❓Watermarking LLMs—Challenges and Opportunities in Electronic Design Automation | IEEE 2025 | This paper reviews watermarking techniques for LLMs in electronic design automation, tackling risks like IP piracy and vulnerabilities in generated RTL code, while exploring opportunities and challenges for robust hardware detection. | [ieee](https://ieeexplore.ieee.org/document/11125763) |
|  |           |  |  |
#### 2.5 📝Parameter Vector Injection:

Focus: Embedding watermarks directly into LLM weights without retraining, preserving functionality for IP tracing.

Methods: This vector embeds watermarks directly into model weights during fine-tuning, enabling transferable ownership signals across model variants without per-instance retraining.

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ✅ Fingerprint Vector: Enabling Scalable and Efficient Model Fingerprint Transfer via Vector Addition | 2025 | Fingerprint Vector enables scalable backdoor fingerprint transfer from base LLMs to downstream models via parameter delta addition, achieving high effectiveness, harmlessness, and robustness. | [arXiv](https://arxiv.org/pdf/2409.08846), [Code](https://github.com/Xuzhenhua55/Fingerprint-Vector) |
| ❓Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes | PETs 2025 | This paper proposes a robust white-box watermarking scheme for large language models, integrating error correction codes with weight permutations to embed and extract identifiers efficiently while resisting adaptive attacks and preserving performance. | [pdf](https://www.petsymposium.org/popets/2025/popets-2025-0126.pdf) |
|  |           |  |  |

### 3 📊 Related Survey

| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ❓Copyright Protection for Large Language Models: A Survey of Methods, Challenges, and Trends | 2025 | This survey reviews LLM copyright protection, distinguishing text watermarking from model fingerprinting, categorizing techniques, introducing transfer/removal methods, evaluation metrics, and future challenges for intellectual property safeguarding. | [arXiv](https://arxiv.org/abs/2508.11548), [Code](https://github.com/Xuzhenhua55/awesome-llm-copyright-protection) |
| ❓SoK: Large Language Model Copyright Auditing via Fingerprinting | 2025 | This SoK provides a unified framework and taxonomy for LLM fingerprinting in copyright auditing, introduces the LEAFBENCH benchmark with 149 models and 13 post-development techniques, and evaluates 8 state-of-the-art methods to reveal their strengths, limitations, and future directions. | [arXiv](https://arxiv.org/abs/2508.19843), [Code](https://github.com/shaoshuo-ss/LeaFBench) |
| ❓Watermarking Large Language Models and the Generated Content: Opportunities and Challenges | IEEE 2025 | This survey examines watermarking techniques for LLMs and generated content, tackling IP protection, robustness against attacks, domain-specific applications, hardware acceleration, and future directions for ethical AI use. | [arXiv](https://arxiv.org/abs/2410.19096) |
| ❓❓Copyright Protection for Large Language Models: A Survey of Methods in Watermarking and Fingerprinting | 2025       |  The survey provides a comprehensive overview of watermarking and fingerprinting techniques for LLMs, addressing intellectual property challenges amid their rapid evolution and widespread use. | [arXiv](https://arxiv.org/abs/2508.11548),[Code](https://github.com/Xuzhenhua55/awesome-llm-copyright-protection) |
| ❓❓Watermarking for Large Language Models: A Survey | MDPI 2025 | Yang et al. (2025) build on prior surveys, such as Liu et al. (2024), which focused on text watermarking, by offering a comprehensive overview of LLM watermarking techniques. Their contribution lies in a detailed taxonomy of training-free and training-based methods, providing insights into algorithmic trade-offs and practical deployment strategies for securing LLM outputs. | [MDPI](https://www.mdpi.com/2227-7390/13/9/1420),[Code](https://github.com/THU-BPM/WaterSeeker) |
|  | |  |  |

### 4 🎭other


| 📄Paper Title                              | 📅Date & Publication | 💡TL;DR                                                                 | 🔗Links            |
|:------------------------------------------|:--------------------:|:---------------------------------------------------------------------|:------------------:|
| ❌ Learnable Fingerprints for Large Language Models | 学生毕业论文 |  | [arXiv](#), [Code](#) |
| ❌Fragile Fingerprinting to Validate Training Data for Large Language Models | 2025(学位论文) | | [proQuest](https://www.proquest.com/openview/3702070489345c3f8fc0463484e4a87f/1?pq-origsite=gscholar&cbl=18750&diss=y) |
| ❌ LLMs Have Rhythm: Fingerprinting Large Language Models Using Inter-Token Times and Network Traffic Analysis | 2025 |  | [arXiv](https://arxiv.org/pdf/2502.20589v1) |
| ❓❓❓Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification | 2025 | TENSORGUARD is a gradient-based fingerprinting framework that extracts behavioral signatures from LLMs via tensor-layer gradient analysis under random perturbations, enabling pairwise similarity detection and 94% accurate family classification for provenance tracking and license compliance. | [arXiv](https://arxiv.org/abs/2506.01631) |
| ❓❓❓ MEraser: An Effective Fingerprint Erasure Approach for Large Language Models | ACL 2025 | MEraser employs a two-phase fine-tuning strategy with mismatched and clean datasets to erase backdoor-based fingerprints from LLMs, achieving 100% removal while preserving performance using under 1,000 samples. | [arXiv](https://arxiv.org/abs/2506.12551), [Code](https://github.com/JingxuanZhang77/MEraser) |
| ❓❓❓A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly | 2024 |  | [arXiv](https://arxiv.org/pdf/2312.02003), [Code](#) |
| ❓❓LLM Fingerprinting via Semantically Conditioned Watermarks | 2025 |  与Robust LLM Fingerprinting via Domain-Specific Watermarks一篇  | [arXiv](https://arxiv.org/abs/2505.16723)                    |
|  | |  |  |



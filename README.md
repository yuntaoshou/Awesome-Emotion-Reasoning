[stars-img]: https://img.shields.io/github/stars/yuntaoshou/Awesome-Emotion-Reasoning?color=yellow
[stars-url]: https://github.com/yuntaoshou/Awesome-Emotion-Reasoning/stargazers
[fork-img]: https://img.shields.io/github/forks/yuntaoshou/Awesome-Emotion-Reasoning?color=lightblue&label=fork
[fork-url]: https://github.com/yuntaoshou/Awesome-Emotion-Reasoning/network/members
[AKGR-url]: https://github.com/yuntaoshou/Awesome-Emotion-Reasoning

# Large Language Models Meet Emotion Recognition: A Survey [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)![](resources/image8.gif)
This is the summation of all the methods, datasets, and other survey mentioned in our survey 'Large Language Models Meet Emotion Recognition: A Survey' :fire:. Any problems, please contact shouyuntao@stu.xjtu.edu.cn. Any other interesting papers or codes are welcome. If you find this repository useful to your research or work, it is really appreciated to star this repository :heart:.

[![GitHub stars][stars-img]][stars-url] 
[![GitHub forks][fork-img]][fork-url]


## Milestone Papers
|   Date  |       keywords       |      Institute     |                                                                                                        Paper                                                                                                       |
|:-------:|:--------------------:|:------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2017-06 |     Transformers     |       Google       | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                                                  |
| 2018-06 |        GPT 1.0       |       OpenAI       | [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)                                                                             |
| 2018-10 |         BERT         |       Google       | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)                                                                                          |
| 2019-02 |        GPT 2.0       |       OpenAI       | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)                                          |
| 2019-09 |      Megatron-LM     |       NVIDIA       | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)                                                                                      |
| 2019-10 |          T5          |       Google       | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)                                                                                       |
| 2019-10 |         ZeRO         |      Microsoft     | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)                                                                                                       |
| 2020-01 |      Scaling Law     |       OpenAI       | [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)                                                                                                                                    |
| 2020-05 |        GPT 3.0       |       OpenAI       | [Language models are few-shot learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)                                                                                         |
| 2021-01 |  Switch Transformers |       Google       | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)                                                                               |
| 2021-08 |         Codex        |       OpenAI       | [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)                                                                                                                           |
| 2021-08 |   Foundation Models  |      Stanford      | [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf)                                                                                                                        |
| 2021-09 |         FLAN         |       Google       | [Finetuned Language Models are Zero-Shot Learners](https://openreview.net/forum?id=gEZrGCozdqR)                                                                                                                    |
| 2021-10 |          T0          | HuggingFace et al. | [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)                                                                                                              |
| 2021-12 |         GLaM         |       Google       | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf)                                                                                                         |
| 2021-12 |        WebGPT        |       OpenAI       | [WebGPT: Browser-assisted question-answering with human feedback](https://www.semanticscholar.org/paper/WebGPT%3A-Browser-assisted-question-answering-with-Nakano-Hilton/2f3efe44083af91cef562c1a3451eee2f8601d22) |
| 2021-12 |         Retro        |      DeepMind      | [Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)                                         |
| 2021-12 |        Gopher        |      DeepMind      | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)                                                                                                 |
| 2022-01 |          COT         |       Google       | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)                                                                                                      |
| 2022-01 |         LaMDA        |       Google       | [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)                                                                                                                             |
| 2022-01 |        Minerva       |       Google       | [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)                                                                                                                   |
| 2022-01 |  Megatron-Turing NLG |  Microsoft&NVIDIA  | [Using Deep and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/pdf/2201.11990.pdf)                                                                         |
| 2022-03 |      InstructGPT     |       OpenAI       | [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)                                                                                                        |
| 2022-04 |         PaLM         |       Google       | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)                                                                                                                              |
| 2022-04 |      Chinchilla      |      DeepMind      | [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)                             |
| 2022-05 |          OPT         |        Meta        | [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068.pdf)                                                                                                                          |
| 2022-05 |          UL2         |       Google       | [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1)                                                                                                                                         |
| 2022-06 |  Emergent Abilities  |       Google       | [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD)                                                                                                                            |
| 2022-06 |       BIG-bench      |       Google       | [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://github.com/google/BIG-bench)                                                                                |
| 2022-06 |        METALM        |      Microsoft     | [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)                                                                                                                             |
| 2022-09 |        Sparrow       |      DeepMind      | [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/pdf/2209.14375.pdf)                                                                                                       |
| 2022-10 |     Flan-T5/PaLM     |       Google       | [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)                                                                                                                              |
| 2022-10 |       GLM-130B       |      Tsinghua      | [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/pdf/2210.02414.pdf)                                                                                                                              |
| 2022-11 |         HELM         |      Stanford      | [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110.pdf)                                                                                                                                     |
| 2022-11 |         BLOOM        |     BigScience     | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/pdf/2211.05100.pdf)                                                                                                            |
| 2022-11 |       Galactica      |        Meta        | [Galactica: A Large Language Model for Science](https://arxiv.org/pdf/2211.09085.pdf)                                                                                                                              |
| 2022-12 |        OPT-IML       |        Meta        | [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/pdf/2212.12017)                                                                                   |
| 2023-01 | Flan 2022 Collection |       Google       | [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf)                                                                                           |
| 2023-02 |         LLaMA        |        Meta        | [LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)                                                            |
| 2023-02 |       Kosmos-1       |      Microsoft     | [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/abs/2302.14045)                                                                                                         |
| 2023-03 |        LRU        |       DeepMind       | [Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)                                                                                                                                          |
| 2023-03 |        PaLM-E        |       Google       | [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io)                                                                                                                                          |
| 2023-03 |         GPT 4        |       OpenAI       | [GPT-4 Technical Report](https://openai.com/research/gpt-4)                                                                                                                                                        |
| 2023-04 |        LLaVA        | UW–Madison&Microsoft | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)                                                                                                                                                     | 
| 2023-04 |        Pythia        |  EleutherAI et al. | [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)                                                                                                |
| 2023-05 |       Dromedary      |     CMU et al.     | [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047)                                                                                 |
| 2023-05 |        PaLM 2        |       Google       | [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)                                                                                                                                  |
| 2023-05 |         RWKV         |       Bo Peng      | [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)                                                                                                                                 |
| 2023-05 |          DPO         |      Stanford      | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)                                                                                             |
| 2023-05 |          ToT         |  Google&Princeton  | [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601.pdf)  |                                                                                                  
| 2023-07 |        LLaMA2       |        Meta        | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)                                                                                                  |
| 2023-08 |        Qwen-VL       |        Alibaba        | [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)                                                                                                 |
| 2023-10 |      Mistral 7B      |       Mistral      | [Mistral 7B](https://arxiv.org/pdf/2310.06825.pdf)                                                                                                                                                                                                 |   
| 2023-11 |      Qwen-Audio      |       Alibaba      | [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](https://arxiv.org/pdf/2311.07919)                                                                                                   |  
| 2023-12 |         Mamba        |    CMU&Princeton   | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)                                                                                                            |
| 2024-01 |         DeepSeek-v2        |      DeepSeek     | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)                                                                                                                          |
| 2024-02 |         OLMo        |      Ai2     | [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)                                                                                                                                                                                                 |
| 2024-05 |         Mamba2        |      CMU&Princeton     | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)                                                                                                 |
| 2024-05 |         Llama3        |      Meta     | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)                                                                                                                                                                                                 |
| 2024-06 |         FineWeb         |      HuggingFace     | [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)                                                                                                 |
| 2024-07 |         Qwen2-Audio         |      Alibaba     | [Qwen2-Audio Technical Report](https://arxiv.org/pdf/2407.10759)                                                                                                                                                                                                 |
| 2024-09 |         OLMoE        |       Ai2     | [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)                                                                                                                                                                                                 |
| 2024-09 |         Qwen2-VL        |       Alibaba     | [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/pdf/2409.12191)                                                                                                 |
| 2024-10 |        Janus        |       DeepSeek     | [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/pdf/2410.13848)                                                                                                 |
| 2024-11 |        JanusFlow        |       DeepSeek     | [JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](https://arxiv.org/pdf/2411.07975)                                                                                                 |
| 2024-12 |         Qwen2.5        |      Alibaba     | [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)                                                                                                                                                                                                 |
| 2024-12 |         DeepSeek-V3        |      DeepSeek     | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v1)                                                                                                                                                                                                 |
| 2024-12 |         QVQ        |      Alibaba     | [QVQ: To See the World with Wisdom](https://qwenlm.github.io/blog/qvq-72b-preview/)                                                                                                                                                                                                 |
| 2024-12 |         DeepSeek-VL2        |      DeepSeek     | [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding](https://arxiv.org/pdf/2412.10302)                                                                                                                                                                                                 |
| 2025-01 |         DeepSeek-R1        |      DeepSeek     | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)                                                                                                 |
| 2025-01 |         Janus-Pro        |      DeepSeek     | [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/pdf/2501.17811)                                                                                                 |
| 2025-02 |         Qwen2.5-VL        |      Alibaba     | [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923)                                                                                                                                                                                                 |
| 2025-03 |         Qwen2.5-Omni        |      Alibaba     | [Qwen2.5-Omni Technical Report](https://arxiv.org/pdf/2503.20215)                                                                                                                                                                                                 |
| 2025-03 |         QwQ        |      Alibaba     | [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)                                                                                                 |
| 2025-05 |         Qwen3         |      Alibaba     | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) |

## Open LLM
<summary>DeepSeek</summary>

  - [DeepSeek-Math-7B](https://huggingface.co/collections/deepseek-ai/deepseek-math-65f2962739da11599e441681)
  - [DeepSeek-Coder-1.3|6.7|7|33B](https://huggingface.co/collections/deepseek-ai/deepseek-coder-65f295d7d8a0a29fe39b4ec4)
  - [DeepSeek-VL-1.3|7B](https://huggingface.co/collections/deepseek-ai/deepseek-vl-65f295948133d9cf92b706d3)
  - [DeepSeek-MoE-16B](https://huggingface.co/collections/deepseek-ai/deepseek-moe-65f29679f5cf26fe063686bf)
  - [DeepSeek-v2-236B-MoE](https://arxiv.org/abs/2405.04434)
  - [DeepSeek-Coder-v2-16|236B-MOE](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
  - [DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)
  - [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
  - [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
  - [DeepSeek-R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)
  - [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
  - [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
  - [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
  - [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
  - [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
  - [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
  - [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
  - [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)
  - [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
  - [DeepSeek-V2-Chat-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628)
  - [DeepSeek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)
  - [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)
  - [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
  - [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)
  - [DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)
  - [DeepSeek-V2.5-1210](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210)
  - [DeepSeek-V3.1-Base ](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base)
  - [DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
  - [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus)
  - [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)
  - [DeepSeek-V3.2-Exp-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp-Base)
  - [Deepseek-Vl2-Tiny](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)
  - [Deepseek-Vl2-Small](https://huggingface.co/deepseek-ai/deepseek-vl2-small)
  - [Deepseek-Vl2](https://huggingface.co/deepseek-ai/deepseek-vl2)
  - [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
  - [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
  - [Janus-1.3B](https://huggingface.co/deepseek-ai/Janus-1.3B)
  - [JanusFlow-1.3B](https://huggingface.co/deepseek-ai/JanusFlow-1.3B)
  - 
  
<summary>Alibaba</summary>

  - [Qwen-1.8B|7B|14B|72B](https://huggingface.co/collections/Qwen/qwen-65c0e50c3f1ab89cb8704144)
  - [Qwen1.5-0.5B|1.8B|4B|7B|14B|32B|72B|110B|MoE-A2.7B](https://qwenlm.github.io/blog/qwen1.5/)
  - [Qwen2-0.5B|1.5B|7B|57B-A14B-MoE|72B](https://qwenlm.github.io/blog/qwen2)
  - [Qwen2.5-0.5B|1.5B|3B|7B|14B|32B|72B](https://qwenlm.github.io/blog/qwen2.5/)
  - [CodeQwen1.5-7B](https://qwenlm.github.io/blog/codeqwen1.5/)
  - [Qwen2.5-Coder-1.5B|7B|32B](https://qwenlm.github.io/blog/qwen2.5-coder/)
  - [Qwen2-Math-1.5B|7B|72B](https://qwenlm.github.io/blog/qwen2-math/)
  - [Qwen2.5-Math-1.5B|7B|72B](https://qwenlm.github.io/blog/qwen2.5-math/)
  - [Qwen-VL-7B](https://huggingface.co/Qwen/Qwen-VL)
  - [Qwen2-VL-2B|7B|72B](https://qwenlm.github.io/blog/qwen2-vl/)
  - [Qwen2-Audio-7B](https://qwenlm.github.io/blog/qwen2-audio/)
  - [Qwen2.5-VL-3|7|72B](https://qwenlm.github.io/blog/qwen2.5-vl/)
  - [Qwen2.5-1M-7|14B](https://qwenlm.github.io/blog/qwen2.5-1m/)
  - [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
  - [Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)

<summary>Meta</summary>

  - [Llama 3.2-1|3|11|90B](https://llama.meta.com/)
  - [Llama 3.1-8|70|405B](https://llama.meta.com/)
  - [Llama 3-8|70B](https://llama.meta.com/llama3/)
  - [Llama 2-7|13|70B](https://llama.meta.com/llama2/)
  - [Llama 1-7|13|33|65B](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
  - [OPT-1.3|6.7|13|30|66B](https://arxiv.org/abs/2205.01068)

<summary>Mistral AI</summary>

  - [Codestral-7|22B](https://mistral.ai/news/codestral/)
  - [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)
  - [Mixtral-8x7B](https://mistral.ai/news/mixtral-of-experts/)
  - [Mixtral-8x22B](https://mistral.ai/news/mixtral-8x22b/)

<summary>Google</summary>

  - [Gemma2-9|27B](https://blog.google/technology/developers/google-gemma-2/)
  - [Gemma-2|7B](https://blog.google/technology/developers/gemma-open-models/)
  - [RecurrentGemma-2B](https://github.com/google-deepmind/recurrentgemma)
  - [T5](https://arxiv.org/abs/1910.10683)

<summary>Apple</summary>

  - [OpenELM-1.1|3B](https://huggingface.co/apple/OpenELM)

<summary>Microsoft</summary>

  - [Phi1-1.3B](https://huggingface.co/microsoft/phi-1)
  - [Phi2-2.7B](https://huggingface.co/microsoft/phi-2)
  - [Phi3-3.8|7|14B](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

<summary>AllenAI</summary>

  - [OLMo-7B](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)

<summary>xAI</summary>

  - [Grok-1-314B-MoE](https://x.ai/blog/grok-os)

<summary>Cohere</summary>

  - [Command R-35B](https://huggingface.co/CohereForAI/c4ai-command-r-v01)

<summary>01-ai</summary>

  - [Yi-34B](https://huggingface.co/collections/01-ai/yi-2023-11-663f3f19119ff712e176720f)
  - [Yi1.5-6|9|34B](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8)
  - [Yi-VL-6B|34B](https://huggingface.co/collections/01-ai/yi-vl-663f557228538eae745769f3)

<summary>Baichuan</summary>

   - [Baichuan-7|13B](https://huggingface.co/baichuan-inc)
   - [Baichuan2-7|13B](https://huggingface.co/baichuan-inc)


<summary>Nvidia</summary>

   - [Nemotron-4-340B](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct)

<summary>BLOOM</summary>

   - [BLOOMZ&mT0](https://huggingface.co/bigscience/bloomz)

<summary>Zhipu AI</summary>

   - [GLM-2|6|10|13|70B](https://huggingface.co/THUDM)
   - [CogVLM2-19B](https://huggingface.co/collections/THUDM/cogvlm2-6645f36a29948b67dc4eef75)

<summary>OpenBMB</summary>

  - [MiniCPM-2B](https://huggingface.co/collections/openbmb/minicpm-2b-65d48bf958302b9fd25b698f)
  - [OmniLLM-12B](https://huggingface.co/openbmb/OmniLMM-12B)
  - [VisCPM-10B](https://huggingface.co/openbmb/VisCPM-Chat)
  - [CPM-Bee-1|2|5|10B](https://huggingface.co/collections/openbmb/cpm-bee-65d491cc84fc93350d789361)

<summary>RWKV Foundation</summary>

  - [RWKV-v4|5|6](https://huggingface.co/RWKV)minicpm-2b-65d48bf958302b9fd25b698f)

<summary>ElutherAI</summary>

  - [Pythia-1|1.4|2.8|6.9|12B](https://github.com/EleutherAI/pythia)

<summary>Stability AI</summary>

  - [StableLM-3B](https://huggingface.co/stabilityai/stablelm-3b-4e1t)
  - [StableLM-v2-1.6B](https://huggingface.co/stabilityai/stablelm-2-1_6b)
  - [StableLM-v2-12B](https://huggingface.co/stabilityai/stablelm-2-12b)
  - [StableCode-3B](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650)

<summary>BigCode</summary>

  - [StarCoder-1|3|7B](https://huggingface.co/collections/bigcode/%E2%AD%90-starcoder-64f9bd5740eb5daaeb81dbec)
  - [StarCoder2-3|7|15B](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a)


<summary>DataBricks</summary>

  - [MPT-7B](https://www.databricks.com/blog/mpt-7b)
  - [DBRX-132B-MoE](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

<summary>Shanghai AI Laboratory</summary>
  
  - [InternLM2-1.8|7|20B](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c)
  - [InternLM-Math-7B|20B](https://huggingface.co/collections/internlm/internlm2-math-65b0ce88bf7d3327d0a5ad9f)
  - [InternLM-XComposer2-1.8|7B](https://huggingface.co/collections/internlm/internlm-xcomposer2-65b3706bf5d76208998e7477)
  - [InternVL-2|6|14|26](https://huggingface.co/collections/OpenGVLab/internvl-65b92d6be81c86166ca0dde4)

## LLM for emotion recognition
| Model            | Supported Modality | Link                                                                 |
|------------------|--------------------|----------------------------------------------------------------------|
| A Multi-Modal Model with In-Context Instruction Tuning        | Video, Text        | [GitHub](https://github.com/Luodian/Otter)                            |
| Videochat: Chat-centric video understanding    | Video, Text        | [GitHub](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat) |
| Mvbench: A comprehensive multi-modal video understanding benchmark   | Video, Text        | [GitHub](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) |
| Video-llava: Learning united visual representation by alignment before projection  | Video, Text        | [GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA)                |
| Video-llama: An instruction-tuned audio-visual language model for video understanding  | Video, Text        | [GitHub](https://github.com/DAMO-NLP-SG/Video-LLaMA)                  |
| Video-chatgpt: Towards detailed video understanding via large vision and language models | Video, Text        | [GitHub](https://github.com/mbzuai-oryx/Video-ChatGPT)                |
| Llama-vid: An image is worth 2 tokens in large language models     | Video, Text        | [GitHub](https://github.com/dvlab-research/LLaMA-VID)                 |
| mplug-owl: Modularization empowers large language models with multimodality    | Video, Text        | [GitHub](https://github.com/X-PLUG/mPLUG-Owl)                        |
| Chat-univi: Unified visual representation empowers large language models with image and video understanding   | Video, Text        | [GitHub](https://github.com/PKU-YuanGroup/Chat-UniVi)                |
| Salmonn: Towards generic hearing abilities for large language models | Audio, Text   | [GitHub](https://github.com/bytedance/SALMONN)                       |
| Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models   | Audio, Text        | [GitHub](https://github.com/QwenLM/Qwen-Audio)                       |
| Secap: Speech emotion captioning with large language model  | Audio, Text        | [GitHub](https://github.com/thuhcsi/SECap)                           |
| Onellm: One framework to align all modalities with language       | Audio, Video, Text | [GitHub](https://github.com/csuhan/OneLLM)                           |
| Pandagpt: One model to instruction-follow them all     | Audio, Video, Text | [GitHub](https://github.com/yxuansu/PandaGPT)                        |
| Emotion-llama: Multimodal emotion recognition and reasoning with instruction tuning | Audio, Video, Text | [GitHub](https://github.com/ZebangCheng/Emotion-LLaMA)                |

## Datasets

| Dataset         | Modality | Samples | Description | Emotions | Annotation Manner                |
|:-----------------:|:----------:|:-----------:|:-------------:|:------------:|:----------------------------------:|
| RAF-DB      | I        | 29,672    | ✗           | 7          | Human                            |
| AffectNet   | I        | 450,000   | ✗           | 8          | Human                            |
| EmoDB        | A        | 535       | ✗           | 7          | Human                            |
| MSP-Podcast | A        | 73,042    | ✗           | 8          | Human                            |
| DFEW        | V        | 11,697    | ✗           | 7          | Human                            |
| FERV39k     | V        | 38,935    | ✗           | 7          | Human                            |
| MER2023     | A,V,T    | 5,030     | ✗           | 6          | Human                            |
| MELD        | A,V,T    | 13,708    | ✗           | 7          | Human                            |
| EmoViT     | I        | 51,200    | ✓           | 988        | Model                            |
| MERR-Coarse | A,V,T    | 28,618    | ✓           | 113        | Model                            |
| MAFW        | A,V,T    | 10,045    | ✓           | 399        | Human                            |
| OV-MERD     | A,V,T    | 332       | ✓           | 236        | Human-led+Model-assisted         |
| MERR-Fine   | A,V,T    | 4,487     | ✓           | 484        | Human-led+Model-assisted         |
| MER-Caption      | A,V,T    | 115,595   | ✓           | 2,932      | Model-led+Human-assisted         |
| MER-Caption+     | A,V,T    | 31,327    | ✓           | 1,972      | Model-led+Human-assisted         |

| Category            | Dataset       | Chosen Set  | # Samples | Label Description                                      |
|---------------------|---------------|-------------|-----------|---------------------------------------------------------|
| Fine-grained Emotion | OV-MERD+      | All         | 532       | unfixed categories and diverse number of labels per sample |
|  Basic Emotion  | MER2023   | MER-MULTI   | 411       | most likely label among six candidates                  |
| Basic Emotion   | MER2024   | MER-SEMI    | 1,169     | most likely label among six candidates                  |
|   Basic Emotion | IEMOCAP   | Sessions5   | 1,241     | most likely label among four candidates                 |
|Basic Emotion    | MELD      | Test        | 2,610     | most likely label among seven candidates                |
|Sentiment Analysis| CMU-MOSI  | Test        | 686       | sentiment intensity, ranging from [-3, 3]               |
|Sentiment Analysis| CMU-MOSEI | Test        | 4,659     | sentiment intensity, ranging from [-3, 3]               |
|Sentiment Analysis| CH-SIMS   | Test        | 457       | sentiment intensity, ranging from [-1, 1]               |
|Sentiment Analysis| CH-SIMS v2 | Test       | 1,034     | sentiment intensity, ranging from [-1, 1]               |

| Dataset         | Domain     | Dur(hrs) | #labels | Modality | Language | Emotion? | Ego? |
|-----------------|------------|----------|---------|----------|----------|----------|------|
| Large Movie | movie      | -        | 25,000  | T        | EN       | ✗        | ✗    |
| SeMAINE     | dialogue   | 06:30    | 80      | V,A      | EN       | ✓        | ✗    |
| HUMAINE     | diverse    | 04:11    | 50      | V,A      | various  | ✓        | ✗    |
| YouTube     | diverse    | 00:29    | 300     | V,A,T    | various  | ✗        | ✗    |
| SST         | movie      | -        | 11,855  | T        | EN       | ✗        | ✗    |
| ICT-MMMO    | movie      | 13:58    | 340     | V,A,T    | EN       | ✗        | ✗    |
| RECOLA      | dialogue   | 03:50    | 46      | V,A      | FR       | ✓        | ✓    |
| MOUD        | review     | 00:59    | 400     | V,A,T    | ES       | ✗        | ✗    |
| AFEW      | movie      | 02:28    | 1,645   | V,A      | various  | ✓        | ✓    |
| SEWA        | adverts   | 04:39    | 538     | V,A      | EN,DE,EL | ✓        | ✗    |
| Disneyworld | disneyland| 42:00    | 15,000  | V,A,T    | EN       | ✗        | ✓    |
| EGTEA Gaze+ | diverse    | 28:00    | -       | V,A,T    | various  | ✓        | ✓    |
| BEOID       | diverse    | -        | -       | V,A,T    | EN       | ✗        | ✗    |
| Chorus-Ego  | home       | 34:00    | 30,000  | V,A,T    | EN       | ✗        | ✓    |
| EPIC       | kitchen    | 100:00   | 90,000  | V,A,T    | EN       | ✗        | ✓    |
| Ego-4D      | diverse    | 3025:00  | 74000   | V,A,T    | various  | ✗        | ✓    |
| \(E^3\)   | diverse    | 71:41    | 81,248  | V,A,T    | various  | ✓        | ✓    |


## Other surveys
| Paper | Url | Source | 
| :---- | :----: | :----: 
| Mm-llms: Recent advances in multimodal large language models | [[paper]](https://arxiv.org/pdf/2401.13601) | [[source]](-) |
| Efficient multimodal large language models: A survey | [[paper]](https://arxiv.org/pdf/2405.10739) | [[source]](https://github.com/swordlidev/Efficient-Multimodal-LLMs-Survey) |
| Hallucination of multimodal large language models: A survey | [[paper]](https://arxiv.org/pdf/2404.18930) | [[source]](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| A survey on benchmarks of multimodal large language models | [[paper]](https://arxiv.org/pdf/2408.08632) | [[source]](https://github.com/swordlidev/Evaluation-Multimodal-LLMs-Survey) |
| A comprehensive survey of large language models and multimodal large language models in medicine | [[paper]](https://arxiv.org/pdf/2405.08603) | - |
| Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning | [[paper]](https://arxiv.org/pdf/2401.06805) | - |
| How to bridge the gap between modalities: A comprehensive survey on multimodal large language model | [[paper]](https://arxiv.org/pdf/2311.07594) | - |
| A Comprehensive Overview of Large Language Models | [[paper]](https://arxiv.org/pdf/2307.06435) | - |
| A review of multi-modal large language and vision models | [[paper]](https://arxiv.org/pdf/2404.01322) | - |
| Large language models meet nlp: A survey | [[paper]](https://arxiv.org/pdf/2405.12819) | - |
| Efficient large language models: A survey | [[paper]](https://arxiv.org/pdf/2312.03863) | [[source]](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey) |

## Acknowledgement :heart:
Thanks to [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuntaoshou/Awesome-Emotion-Reasoning&type=Date)](https://star-history.com/#yuntaoshou/Awesome-Emotion-Reasoning&Date)

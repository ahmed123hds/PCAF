# Suggested arXiv Metadata

Title: Parallel Causal Associative Fields: Gated Sparse Memory for Long-Context Language Modeling

Authors: Ahmed

Primary category: cs.LG

Optional cross-list: cs.CL

Comments: Preprint. 15 pages in the conference-formatted version; experiments include WikiText-103, PG-19, WikiText-2, TPU v4-32, and RTX 3060 evaluations. Code: https://github.com/ahmed123hds/PCAF

## Abstract

Transformers achieve strong language modeling performance by providing direct token-to-token communication paths, but causal self-attention scales quadratically with context length. Recurrent and state-space models reduce this cost, yet compress history into sequentially updated fixed-size states. This paper studies a third primitive: a parallel content-addressed memory over causal successor records. The proposed Parallel Causal Associative Field (PCAF) writes local records from a context window into hash buckets, retrieves a bounded candidate set for the current query, forms a sparse cache distribution over successor tokens, and mixes that cache with a parametric local language model through a learned gate. The resulting model maintains sparse long-context access while avoiding a single fixed recurrent state bottleneck. We evaluate PCAF under full autoregressive pretraining on WikiText-103 and PG-19 using a distributed Google Cloud TPU v4-32 pod. At 303M parameters and context length T=2048, PCAF-semantic reaches 36.31 perplexity on WikiText-103 and 52.45 perplexity on PG-19, compared with 47.49 and 53.84 for a matched dense Transformer. PCAF-semantic simultaneously processes 0.61--0.62M tokens/s across the TPU pod, versus 0.43M tokens/s for dense and local attention baselines. Supporting 41M-parameter multi-seed sweeps and single-GPU component ablations confirm that the associative cache, retrieval capacity, and learned gate materially affect the speed-quality trade-off.

## Upload Notes

- Upload the generated `pcaf_arxiv_source.tar.gz`.
- Select `pcaf_arxiv.tex` as the main TeX file if arXiv does not detect it automatically.
- Review the author name and public email before submission.
- Choose the arXiv license in the submission form; it is not encoded in the source package.

# Objective - Qwen3-8B inference server on a single RTX 3090

Serve `Qwen/Qwen3-8B` on one NVIDIA RTX 3090 with the best throughput and
latency trade-off under a realistic concurrent request load, while preserving
the accuracy contract. Build an OpenAI-compatible `/v1/chat/completions` and
`/v1/completions` server.

This is the Qwen3-8B counterpart to the full Llama-3-8B Request Factory task.
The task keeps the same two-axis Pareto objective and end-to-end evaluation
shape, but pins the target model and adapts the concurrency ladder to the
24 GiB target GPU. This is a full serving task, not a CPU or fake-server smoke
task.

The candidate must implement the inference engine from the model contract:
Qwen3 RMSNorm, QK-Norm, GQA attention, RoPE, SwiGLU MLP, KV-cache prefill and
decode, greedy temperature-zero generation, streaming SSE, and bounded
continuous batching. `transformers` may be used for tokenizer/config/weight
loading and reference comparison. It must not replace the candidate serving
engine or the evaluator.

## Objective and canonical metrics

The canonical benchmark is the Request Factory concurrency sweep. It reports
one selected sustainable operating point, and both metrics must come from that
same point:

- `aggregate_throughput`: output tokens/sec, maximize.
- `p99_latency_ms`: p99 end-to-end latency in milliseconds, minimize.

The scalar `perf_metric` is the selected point's `aggregate_throughput` and
`perf_unit` is `tok/s`. A candidate is retained on the Pareto frontier when it
is non-dominated outside the configured 3% noise band.

## Benchmark protocol

Request Factory sends 512 independent requests per point over loopback. Each
request has 256 input tokens and 128 generated tokens, `temperature=0`, and
streaming enabled. The server stays alive across the sweep. The target's 24 GiB
memory budget uses the ladder `1, 2, 4, 8, 16, 32`; an implementation may not
silently reject requests or shorten the workload. OOM, failed requests, or an
unserved point are failures, not an overload classification shortcut.

The benchmark checks health before the sweep, retains every row, confirms a
suspected overload bracket, repeats the selected point and neighbors, and reads
both objective values verbatim from the selected row. It must not mix throughput
from one operating point with latency from another.

## Model and runtime contract

- Model: `Qwen/Qwen3-8B`.
- Hugging Face source revision: `main`, resolved commit
  `b968826d9c46df6066d109eabc6255188de91218`.
- Dtype: BF16 reference path. A different dtype requires a separately recorded
  accuracy comparison and is not silently treated as the same result.
- Hardware: one RTX 3090, 24 GiB VRAM, CUDA device 0.
- Qwen3 chat requests must disable thinking for this serving workload with
  `enable_thinking=false`; the server must honor preformatted prompts without
  applying a second chat-template pass.
- Weights and tokenizer are external artifacts. They are never committed to
  this repository. The accuracy checker drives the running HTTP server and
  must not import the candidate's model.

## Correctness and anti-bypass requirements

The accuracy checker uses fresh sentinel prompts, known-answer prompts, and
repeated temperature-zero prompts. It rejects canned output, prompt echoing,
transport failures, and nondeterministic greedy decoding. The server must run a
real prompt-conditioned Qwen3-8B forward pass and preserve the OpenAI response
and SSE contract. The evaluator and benchmark are trusted task inputs and must
remain unchanged during an optimization round.

The first implementation should prioritize semantic parity with the pinned
Hugging Face reference. Kernel, batching, KV-cache, memory-layout, and CUDA
graph changes are valid only when the accuracy gate remains green.

# qwen3-8b-rtx3090 - full Qwen3 inference task

This bundle adapts the complete Llama-3-8B Request Factory task to a single
RTX 3090 and `Qwen/Qwen3-8B`. It is a full OpenAI-compatible serving objective,
with an accuracy gate plus a throughput/latency concurrency sweep. It is not a
minimal CPU smoke task.

The reference architecture is the generated Qwen3 implementation from the
Qwen3-32B task, paired with the pinned Qwen3-8B configuration. The candidate
must provide its own serving engine and may use `transformers` only for config,
tokenizer, weight loading, and reference semantics.

## External model artifacts

The task uses the public Hugging Face repository `Qwen/Qwen3-8B`. The current
resolved commit is `b968826d9c46df6066d109eabc6255188de91218`; the task metadata
records the revision and expected commit. Set `HF_HOME=/env/huggingface` (the
shared environment file does this) and populate a local model snapshot before
running the server. Do not commit weights.

## Run shape

The full Llama-style Request Factory protocol is retained: 512 requests per
point, 256 input tokens, 128 output tokens, and a concurrency sweep through
1, 2, 4, 8, 16, and 32. The ladder is bounded at 32 because the target is a
24 GiB card; the workload itself is not reduced to a one-request smoke test.

```bash
vibesys --runs-dir /work/vibesys-runs --local \
  --input examples/model-serving/qwen3-8b-rtx3090 \
  --config /env/vibesys/agent.toml --backend cuda
```

The current host environment must still provide PyTorch with CUDA support,
model weights, and a usable VibeSys AgentShim sandbox before a formal run.

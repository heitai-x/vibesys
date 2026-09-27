# Qwen3-8B reference bundle

`reference.py` is the generated Qwen3 model implementation from the official
Qwen3 reference used by the sibling Qwen3 task. `config.json` is the pinned
Qwen3-8B configuration. `meta.json` records the Hugging Face model and
resolved commit. Weights stay outside Git and are loaded from the configured
Hugging Face cache.

The candidate server must implement its own serving path; this reference is
only a semantic source for model behavior and accuracy comparison.

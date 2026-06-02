# Oscar Quant

`oscar_quant` is a Python 3.12+ project for two related but different
quantization jobs:

1. **Runtime OScaR KV-cache quantization** for supported generation models.
2. **Persistent `.safetensors` weight artifacts** using Hugging Face
   Transformers plus TorchAO.

The runtime OScaR path now supports:

| Profile | Model ID | Runtime patch | Loader |
| --- | --- | --- | --- |
| `granite-4.0-1b-base` | `ibm-granite/granite-4.0-1b-base` | `GraniteAttention` / `GraniteMoeHybridAttention` | `AutoModelForCausalLM` |
| `gemma4-e2b` | `google/gemma-4-E2B` | `Gemma4TextAttention` | `AutoModelForImageTextToText` |

The file-producing TorchAO artifact path supports the same profiles.

## Quick Mental Model

OScaR KV-cache quantization and `.safetensors` weight quantization are not the
same output.

Runtime OScaR:

1. Loads the original model weights.
2. Patches attention `forward` methods in memory.
3. Lets Hugging Face `generate(...)` create a normal KV cache.
4. Rotates/processes K/V tensors with OScaR.
5. Quantizes cached K/V tensors after attention uses them.

This returns an **in-memory patched model object**. It does not create a
standalone `.safetensors` file.

Important: the current Granite and Gemma4 runtime patch uses upstream OScaR's
`quantize_prefill(...)` / `quantize_kv_cache(...)` path. That path fake-quantizes
and dequantizes back into floating-point cache tensors, so it can simulate
quantization error without physically packing KV cache storage. Use the
benchmark's `kv_cache_observed_gib` and `kv_cache_storage_note` fields to verify
whether a run changed actual tensor storage.

TorchAO `.safetensors` export:

1. Loads the original model weights.
2. Applies TorchAO weight quantization through Transformers.
3. Saves a Hugging Face-compatible model directory.
4. Writes one or more `.safetensors` files.

This creates **persistent quantized model weight files**. It is not OScaR KV
cache compression.

## Where The Gemma4 OScaR Code Lives

The Gemma4 runtime patch is here:

```text
src/oscar_quant/gemma4_patch.py
```

The high-level Gemma4 loader is here:

```text
src/oscar_quant/loader.py
```

Use this Python API:

```python
from oscar_quant import OscarKVConfig, load_oscar_patched_gemma4

patched_gemma = load_oscar_patched_gemma4(
    kv_config=OscarKVConfig(k_bits=2, v_bits=2),
    torch_dtype="auto",
    device_map="auto",
)

print(patched_gemma.patched_attention_layers)
print(patched_gemma.generate_text("Explain KV-cache quantization in one sentence.", max_new_tokens=64))
```

What happens under the hood:

- Source Gemma4 text attention layers own the K/V cache and run full OScaR
  processing plus cache quantization.
- Gemma4 shared-KV layers reuse source-layer K/V states and apply only the
  query-side OScaR rotation, avoiding a double rotation of shared keys.
- Gemma4 full/sliding attention mask routing remains controlled by
  Transformers' Gemma4 model code.

## Setup

From this repo:

```bash
cd oscar-granite-kv-quant
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[artifact]"
```

If you use `uv`:

```bash
cd oscar-granite-kv-quant
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e ".[artifact]"
```

Check the package import without downloading a model:

```bash
python -c "from oscar_quant import OscarKVConfig; print(OscarKVConfig())"
```

## Install OScaR Runtime Dependency

Runtime KV-cache patching needs upstream OScaR:

```bash
bash scripts/install_oscar_dependency.sh
```

That script clones upstream OScaR into `third_party/OScaR-KV-Quant`, initializes
its submodules, installs the CUDA PyTorch wheel used by OScaR, and installs
OScaR editable into the active environment.

Use a Linux machine with an NVIDIA CUDA GPU for the OScaR runtime path because
upstream OScaR builds CUDA extensions.

## Run OScaR Generation

Granite:

```bash
oscar-generate \
  --profile granite-4.0-1b-base \
  --prompt "Explain KV-cache quantization in one paragraph." \
  --max-new-tokens 128 \
  --k-bits 2 \
  --v-bits 2
```

Gemma4-E2B:

```bash
oscar-generate \
  --profile gemma4-e2b \
  --prompt "Explain KV-cache quantization in one paragraph." \
  --max-new-tokens 128 \
  --k-bits 2 \
  --v-bits 2
```

Expected stderr:

```text
patched_gemma4-e2b_attention_layers=<positive integer>
```

Expected stdout:

```text
<generated model text>
```

## Python API For Runtime OScaR

Granite:

```python
from oscar_quant import OscarKVConfig, load_oscar_patched_granite

patched_granite = load_oscar_patched_granite(
    kv_config=OscarKVConfig(k_bits=2, v_bits=2),
    torch_dtype="auto",
    device_map="auto",
)

text = patched_granite.generate_text(
    "Explain OScaR KV-cache quantization in one sentence.",
    max_new_tokens=64,
)
print(text)
```

Gemma4-E2B:

```python
from oscar_quant import OscarKVConfig, load_oscar_patched_gemma4

patched_gemma = load_oscar_patched_gemma4(
    kv_config=OscarKVConfig(k_bits=2, v_bits=2),
    torch_dtype="auto",
    device_map="auto",
)

text = patched_gemma.generate_text(
    "Explain OScaR KV-cache quantization in one sentence.",
    max_new_tokens=64,
)
print(text)
```

The returned wrapper has:

- `model_id`: selected Hugging Face model id.
- `model`: patched Hugging Face model object.
- `tokenizer` for Granite or `processor` for Gemma4.
- `kv_config`: validated `OscarKVConfig`.
- `patched_attention_layers`: number of attention modules patched.

## Create Quantized `.safetensors`

Install the artifact extra first:

```bash
python -m pip install -e ".[artifact]"
```

Granite INT4:

```bash
oscar-quantize-weights \
  --profile granite-4.0-1b-base \
  --output-dir artifacts/granite-4.0-1b-base-int4 \
  --quantization int4_weight_only \
  --group-size 128 \
  --dtype bfloat16 \
  --device-map auto \
  --max-shard-size 10GB
```

Gemma4-E2B INT4:

```bash
oscar-quantize-weights \
  --profile gemma4-e2b \
  --output-dir artifacts/gemma-4-e2b-int4 \
  --quantization int4_weight_only \
  --group-size 128 \
  --dtype bfloat16 \
  --device-map auto \
  --max-shard-size 10GB
```

`model-quantize-weights` is the same exporter under a generic alias.

Available quantization methods:

- `int4_weight_only`
- `int8_weight_only`
- `int8_dynamic_activation_int8_weight`

The exporter prints a JSON report like:

```json
{
  "profile": "gemma4-e2b",
  "model_id": "google/gemma-4-E2B",
  "auto_model_class": "image-text-to-text",
  "output_dir": "/absolute/path/artifacts/gemma-4-e2b-int4",
  "quantization": "int4_weight_only",
  "group_size": 128,
  "dtype": "bfloat16",
  "device_map": "auto",
  "max_shard_size": "10GB",
  "safetensors_files": [
    {
      "path": "model.safetensors",
      "size_bytes": 123456789
    }
  ]
}
```

Depending on `--max-shard-size`, the model may be saved as one file:

```text
model.safetensors
```

Or multiple shards:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

Both forms are normal Hugging Face `save_pretrained` outputs.

## Load A Saved Artifact

Granite:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

artifact_dir = "artifacts/granite-4.0-1b-base-int4"

tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForCausalLM.from_pretrained(
    artifact_dir,
    device_map="auto",
    torch_dtype="auto",
)
```

Gemma4-E2B:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

artifact_dir = "artifacts/gemma-4-e2b-int4"

processor = AutoProcessor.from_pretrained(artifact_dir)
model = AutoModelForImageTextToText.from_pretrained(
    artifact_dir,
    device_map="auto",
    torch_dtype="auto",
)
```

TorchAO must be installed in the environment that loads the quantized artifact.

## Python API For `.safetensors`

```python
from oscar_quant import ArtifactQuantizationConfig, quantize_model_to_safetensors

report = quantize_model_to_safetensors(
    ArtifactQuantizationConfig(
        profile="gemma4-e2b",
        output_dir="artifacts/gemma-4-e2b-int4",
        quantization="int4_weight_only",
        group_size=128,
        dtype="bfloat16",
        device_map="auto",
        max_shard_size="10GB",
    )
)

print(report.output_dir)
print(report.safetensors_files)
```

`quantize_granite_to_safetensors(...)` remains available as a
backward-compatible Granite wrapper.

## Benchmark Granite

The baseline benchmark currently compares Granite vanilla generation against
Granite OScaR generation:

```bash
oscar-baseline \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --k-bits 2 \
  --v-bits 2
```

The command prints JSON with latency, generated tokens, tokens/sec, generated
text, CUDA peak memory when CUDA is available, and OScaR cache storage
diagnostics. If `kv_cache_storage_note` says physical tensor storage is
unchanged, the run is using fake quantize/dequantize rather than packed KV cache
storage.

Use `--kv-cache-mode` to make the intended cache path explicit:

```bash
oscar-baseline \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --k-bits 2 \
  --v-bits 2 \
  --kv-cache-mode fake
```

Current Granite/Gemma4 support is `fake` only. Passing `--kv-cache-mode packed`
fails fast because this repo does not yet contain a model-family-specific packed
KV cache implementation for Granite or Gemma4.

## Remote Long-Context Validation

On the A800 machine, install the package and runtime dependency first:

```bash
python -m pip install -e ".[artifact]"
bash scripts/install_oscar_dependency.sh
```

Download the supported model snapshots into a local Hugging Face cache:

```bash
python scripts/download_models.py --cache-dir checkpoints/huggingface
```

If Gemma4 access is gated in your Hugging Face account, login first or pass a
token:

```bash
huggingface-cli login
# or
python scripts/download_models.py --cache-dir checkpoints/huggingface --token "$HF_TOKEN"
```

Run the default long-context matrix for Granite and Gemma4 at 1K, 2K, and 8K
tokens:

```bash
./scripts/run_long_context_matrix.sh
```

The default matrix runs:

- Baseline: `bf16`, `fp16`
- OScaR fake KV path: `int8`, `int4`, `int2`
- Contexts: `1024`, `2048`, `8192`
- New tokens: `32`

Customize the run with environment variables:

```bash
CONTEXTS="1024 2048 8192" \
PROFILES="granite-4.0-1b-base gemma4-e2b" \
BASELINE_PRECISIONS="bf16 fp16" \
OSCAR_PRECISIONS="int8 int4 int2" \
KV_CACHE_MODE="fake" \
MAX_NEW_TOKENS=32 \
RESULT_ROOT="results/a800_long_context" \
./scripts/run_long_context_matrix.sh
```

Run one case directly:

```bash
PYTHONPATH=src python scripts/run_long_context_case.py \
  --profile granite-4.0-1b-base \
  --run-type oscar \
  --precision int2 \
  --kv-cache-mode fake \
  --context-target 8192 \
  --max-new-tokens 32 \
  --output-json results/long_context/raw/granite_oscar_int2_fake_8192.json
```

The matrix writes raw JSON under `results/long_context/raw/` and a CSV summary
at `results/long_context/summary.csv`. Important fields:

- `nvidia_smi_peak_used_gib`: process-visible GPU peak from `nvidia-smi`.
- `torch_peak_allocated_gib`: PyTorch allocated peak during generation.
- `kv_theoretical_bf16_gib`: formula-based bf16/fp16 KV size.
- `kv_theoretical_quantized_gib`: formula-based int8/int4/int2 KV size.
- `kv_observed_tensor_gib`: actual OScaR cache tensor bytes observed by the
  runtime patch.
- `kv_cache_storage_note`: tells whether cache tensor dtype/shape/bytes changed.

For the current fake path, expect `kv_cache_storage_note` to report that physical
cache tensor storage is unchanged. That means the run simulated quantization
error but did not physically pack the KV cache.

## Important CLI Options

For runtime OScaR generation:

- `--profile`: `granite-4.0-1b-base` or `gemma4-e2b`.
- `--model-id`: optional override for the profile model id.
- `--k-bits` and `--v-bits`: key/value cache quantization bit widths.
- `--k-groupsize` and `--v-groupsize`: KV-cache quantization group sizes.
- `--max-new-tokens`: number of tokens to generate.
- `--temperature`: `0.0` means greedy decoding; values above zero enable
  sampling.
- `--device-map`: passed to Hugging Face model loading. Defaults to `auto`.
- `--dtype`: one of `auto`, `bfloat16`, `float16`, or `float32`.

For `.safetensors` artifacts:

- `--profile`: `granite-4.0-1b-base` or `gemma4-e2b`.
- `--model-id`: optional override for the profile model id.
- `--auto-model-class`: optional override, `causal-lm` or
  `image-text-to-text`.
- `--output-dir`: where to save the quantized model directory.
- `--quantization`: `int4_weight_only`, `int8_weight_only`, or
  `int8_dynamic_activation_int8_weight`.
- `--group-size`: group size for INT4 weight-only quantization.
- `--max-shard-size`: passed to `save_pretrained`. Use a large value like
  `10GB` if you want a single `model.safetensors` when possible.

## Project Layout

```text
src/oscar_quant/
  __init__.py        Public package exports
  artifact.py        Shared quantized .safetensors weight artifact exporter
  benchmark.py       Baseline vs OScaR benchmark CLI for Granite
  cli.py             Runtime OScaR generation CLI for Granite and Gemma4
  config.py          Pydantic OScaR KV config
  gemma4_patch.py    Runtime Gemma4 text attention patch
  granite_patch.py   Runtime Granite attention patch
  kv_cache_utils.py  Shared OScaR/cache helper functions
  loader.py          High-level patched model loaders
  models.py          Shared Granite/Gemma model profiles
  schemas.py         Pydantic benchmark result schemas

scripts/
  download_models.py
  install_oscar_dependency.sh
  run_long_context_case.py
  run_long_context_matrix.sh
  summarize_long_context_results.py

tests/
  test_config.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'kv_cache_compression'`

This affects only runtime OScaR generation. Install OScaR:

```bash
bash scripts/install_oscar_dependency.sh
```

### No Gemma4 attention modules were found

Install or upgrade Transformers to a release with Gemma4 support:

```bash
python -m pip install --upgrade "transformers>=5.5"
```

Then make sure the model id is Gemma4, for example:

```bash
oscar-generate --profile gemma4-e2b --prompt "Hello"
```

### Missing `AutoModelForImageTextToText`

Gemma4-E2B uses the image-text-to-text auto class. Upgrade Transformers:

```bash
python -m pip install --upgrade "transformers>=5.5"
```

### `ModuleNotFoundError: No module named 'torchao'`

This affects only `.safetensors` artifact export. Install the artifact extra:

```bash
python -m pip install -e ".[artifact]"
```

### No `.safetensors` files were written

The exporter calls `save_pretrained(..., safe_serialization=True)` and then
checks for `*.safetensors`. If none are found, check whether model saving failed
earlier in the logs and confirm that `safetensors` is installed.

### CUDA or TorchAO quantization fails

Check that your PyTorch, TorchAO, CUDA, and driver versions are compatible. The
TorchAO backend you choose may have hardware-specific requirements.

### Out of memory while exporting

Try:

- Use `--device-map auto`.
- Use `--dtype float16` or `--dtype bfloat16`.
- Close other GPU workloads.
- Export on a larger GPU machine.
- Try Granite before Gemma if you only want to test the pipeline.

## Source References

- IBM Granite 4.0 1B Base model card:
  https://huggingface.co/ibm-granite/granite-4.0-1b-base
- Google Gemma4-E2B model card:
  https://huggingface.co/google/gemma-4-E2B
- Hugging Face Gemma4 docs:
  https://huggingface.co/docs/transformers/v5.8.1/model_doc/gemma4
- Hugging Face Gemma4 model source:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py
- Hugging Face TorchAO quantization docs:
  https://huggingface.co/docs/transformers/main/quantization/torchao
- OScaR-KV-Quant:
  https://github.com/ZunhaiSu/OScaR-KV-Quant

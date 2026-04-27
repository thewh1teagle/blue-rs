# blue-rs

Rust ONNX inference for BlueTTS.

The library expects phonemes at the inference boundary. Text phonemization is
available as a helper module and examples, but `BlueTts::synthesize` itself does
not phonemize.

## Models

From this directory, examples use the parent repo models:

```bash
../onnx_models-int8
../voices/female1.json
```

For the embedded example, `renikud.onnx` is included from this crate directory:

```bash
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
```

## Run

Basic phoneme-only inference:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --example basic
```

Self-contained text example with embedded ONNX model bytes:

```bash
SDKROOT=$(xcrun --show-sdk-path) cargo run --release --example embedded -- \
  --language he "שלום עולם" ../examples/out/embedded-he.wav
```

Supported `--language` values for the phonemizer helper are `he`, `en`, `es`,
`de`, and `it`.

On macOS, `SDKROOT=...` may be needed for `espeak-rs-sys`/bindgen to find system
headers.

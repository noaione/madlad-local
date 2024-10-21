# madlad-local

Run MADLAD-400 3b-MT model locally on your computer, created since I want to try out MADLAD-400 for myself (and also to have local MT models).

Utilize HuggingFace Hub to download the model locally and Candle as the inference engine.

## Requirements
- Windows 10/11
- Visual Studio 2022 + MSVC Toolkit

If you want to do GPU computing, you need:
- CUDA 12.4
- cuDNN 9.2 (for CUDA 12.x)

## Installations
1. `git clone https://github.com/noaione/madlad-local.git`
2. Run `cargo build --release`
   - If you want to use CUDA+cuDNN, use `cargo build --release --features cuda-compute`
3. Run `./target/release/madlad.exe`

## Usages

You can use madlad like this:
```ps1
madlad.exe --target ja "The quick brown fox jumps over the lazy dog"
```

Which should output something like this:
```ps1
✨✨✨✨ MADLAD-400-3B-MT ✨✨✨✨

[?] Loading config from [REDACTED]\models--jbochi--madlad400-3b-mt\snapshots\bb45f1851bf13a0b8f4fcd9ecadf2cdb7cf22439\config.json
[?] Loading tokenizer from [REDACTED]\models--jbochi--madlad400-3b-mt\snapshots\bb45f1851bf13a0b8f4fcd9ecadf2cdb7cf22439\tokenizer.json
[?] Loading weights from [REDACTED]\models--jbochi--madlad400-3b-mt\snapshots\bb45f1851bf13a0b8f4fcd9ecadf2cdb7cf22439\model-q4k.gguf
[?] Loaded config: [REDACTED]\models--jbochi--madlad400-3b-mt\snapshots\bb45f1851bf13a0b8f4fcd9ecadf2cdb7cf22439\config.json
[?] Loaded tokenizer: [REDACTED]\models--jbochi--madlad400-3b-mt\snapshots\bb45f1851bf13a0b8f4fcd9ecadf2cdb7cf22439\tokenizer.json
[?] Tokenizing input...
[?] Creating input tensor...
[?] Building model...
[?] Creating output tensor...
[?] Creating logits processor...
[?] Encoding input tokens...

[?] Inferencing...
[!] Input: The quick brown fox jumps over the lazy dog
[!] Output: 速い褐色狐は怠け犬を飛び越えた
17 tokens generated (28.29 token/s)
```

MADLAD-400 are able to infer the source language automatically, so you only need to provide the target language.

For full list of supported languages, run `madlad.exe --help`<br />
There's also more configuration that you can set, see `madlad.exe --help`

## License

MIT License

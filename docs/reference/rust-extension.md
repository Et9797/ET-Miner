# Rust extension

The optional `et_miner_rust` extension is built from `rust_ext/` with maturin;
the build recipe is in [Getting started](../getting-started.md). When it is
importable, `et_miner.HAS_RUST` is true and `et_miner.apriori_from_csr` is the
Rust implementation; otherwise `apriori_from_csr` is a stub that raises
{py:class}`~et_miner.exceptions.MiningError` with the build command.

## Functions exported to Python

The extension registers these functions (`rust_ext/src/lib.rs`):

```{literalinclude} ../../rust_ext/src/lib.rs
:language: rust
:start-at: "#[pymodule]"
:end-at: "Ok(())"
```

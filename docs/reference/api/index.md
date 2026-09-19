# API reference

The `et_miner` package re-exports the mining API and the capability probes;
`et_miner.core`, `et_miner.streaming`, `et_miner.gpu` and `et_miner.io` hold
the implementation modules.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   et_miner
```

## Capability flags

These module constants are set once at import time from
{py:mod}`et_miner.backends`; autodoc lists only documented data, so they are
declared here.

```{eval-rst}
.. autodata:: et_miner.HAS_GPU

   CuPy is importable, so the GPU modules are usable.

.. autodata:: et_miner.HAS_MULTI_GPU

   More than one CUDA device is visible right now.

.. autodata:: et_miner.HAS_RUST

   The ``et_miner_rust`` extension is importable.
```

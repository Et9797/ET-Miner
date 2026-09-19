# Configuration

```{include} ../../README.md
:start-after: "## Configuration"
:end-before: "## Features"
```

The TOML/environment configuration is {py:class}`et_miner.config.Config`,
loaded by {py:func}`et_miner.config.load_config`.

## All `ET_*` environment variables

The module docstring of `src/et_miner/_env.py` is the single place where
every knob is documented:

```{literalinclude} ../../src/et_miner/_env.py
:language: text
:start-after: Variables:
:end-before: '"""'
```

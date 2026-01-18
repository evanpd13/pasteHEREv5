# newPROJECTv5

## macOS (Apple Silicon) stability notes

On macOS (including 2024 MacBook Pro models), this app defaults to CPU to avoid
intermittent MPS-related crashes. To opt in to MPS acceleration, set:

```bash
export ZERO123_DEVICE=mps
```

If you run into crashes on macOS, use Python 3.11 and re-run `./run.sh`.

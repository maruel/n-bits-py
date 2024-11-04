# n-bits

Algorithms to better understand DNN (deep neural networks) weights.


## Installation

```bash
pip install n_bits
```


## Usage

Analyze Llama-3.2 1B:

```bash
n-bits analyze --hf-repo meta-llama/Llama-3.2-1B
```


### As a Python Package

```python
import huggingface_hub
import safetensors

from n_bits.analyze import graph_histogram

# Load a tensor
local_path = huggingface_hub.hf_hub_download(repo_id="meta-llama/Llama-3.2-1B", filename="model.safetensors")
tensors = safetensors.torch.load_file(local_path)
first_name = next(iter(tensors))
first = tensors[first_name]
graph_histogram(first, first_name)
```

Code coverage:
[![codecov](https://codecov.io/gh/maruel/n-bits-py/graph/badge.svg?token=D54RD4K2OH)](https://codecov.io/gh/maruel/n-bits-py)

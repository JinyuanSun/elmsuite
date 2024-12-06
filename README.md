# elmsuite

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple, unified interface to multiple evolutionary language models (Protein, DNA, and RNA).
This project is heavily inspired by [aisuite](https://github.com/andrewyng/aisuite).

## Installation

```shell
git clone https://github.com/JinyuanSun/elmsuite.git
cd elmsuite && pip install -e .
```
## Set up

To get started, you will need API Keys for the providers of certain models you intend to use. 
Set the API keys.
```shell
export GINKGO_API_KEY="your-ginkgo-api-key"
export BIOLM_API_KEY="your-biolm-api-key"
```

Use the python client to compute protein sequnece embeddings.
```python
import elmsuite as elm
import numpy as np
interface = elm.Interface()
models = ["ginkgo:ginkgo-aa0-650M"]
sequence = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
results = interface.infer.embed.create(model=models[0], sequence=sequence)

embed = np.array(results.choices[0].embedding.content)
print(embed.shape)

# (1280,)
```

Use the python client to generate protein sequence completions from `biolm:progen2-medium`.
```python
import elmsuite as elm

interface = elm.Interface()
models = ["biolm:progen2-medium"]
sequence = "MTYKLILNGKTLKGETTTEAVDAAT"
results = interface.infer.completion.create(model=models[0], prompt=sequence, num_samples=3, max_length=100)

# This API is slow when tested 2024.12.02
```

Single-point mutation-based protein engineering using `ginkgo:aa0-650M`.  

```python
import elmsuite as elm

interface = elm.Interface()
models = ["ginkgo:ginkgo-aa0-650M"]
sequence = "MTYKLILNGKTLKGETTTEAVDAAT<mask>EKVFKQYANDNGVDGEWTYDDATKTFTVTE"
results = interface.infer.completion.create(model=models[0], prompt=sequence)
print(results.choices[0].sequence.content)
```

Evo on together.ai:

```python
import elmsuite as elm
import numpy as np
interface = elm.Interface()
models = ["together:togethercomputer/evo-1-131k-base"]
sequence = "ATG"
results = interface.infer.completion.create(model=models[0], prompt=sequence, max_tokens=100)

# The API is not available on together.ai when tested 2024.12.02. But it should work, they published a Science paper (https://doi.org/10.1126/science.ado9336) based on the model.
```

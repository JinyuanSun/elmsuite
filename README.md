# plmsuite

Simple, unified interface to multiple protein language models.


## Installation

```shell
git clone https://github.com/JinyuanSun/plmsuite.git
cd plmsuite && pip install -e .
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
import plmsuite as plm
import numpy as np
interface = plm.Interface()
models = ["ginkgo:ginkgo-aa0-650M"]
sequence = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
results = interface.infer.embed.create(model=models[0], sequence=sequence)

embed = np.array(results[0]['result']['embedding'])
print(embed.shape)

# (1280,)
```

Use the python client to generate protein sequence completions from `biolm:progen2-medium`.
```python
import plmsuite as plm
import numpy as np
interface = plm.Interface()
models = ["biolm:progen2-medium"]
sequence = "MTYKLILNGKTLKGETTTEAVDAAT"
results = interface.infer.completion.create(model=models[0], prompt=sequence, num_samples=3, max_length=100)

# This API is slow when tested 2024.12.02
```

Single-point mutation-based protein engineering using `ginkgo:aa0-650M`.  

```python
import plmsuite as plm
import numpy as np
interface = plm.Interface()
models = ["ginkgo:ginkgo-aa0-650M"]
sequence = "MTYKLILNGKTLKGETTTEAVDAAT<mask>EKVFKQYANDNGVDGEWTYDDATKTFTVTE"
results = interface.infer.completion.create(model=models[0], prompt=sequence)
print(results[0]["result"]['sequence'])
```

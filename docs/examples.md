# Examples
- Note: Examples here may not be functional, as the API is not available when tested 2024.12.02.

Use the python client to generate protein sequence completions from `biolm:progen2-medium`.
```python
import elmsuite as elm

interface = elm.Interface()
models = ["biolm:progen2-medium"]
sequence = "MTYKLILNGKTLKGETTTEAVDAAT"
results = interface.infer.completion.create(model=models[0], prompt=sequence, num_samples=3, max_length=100)

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

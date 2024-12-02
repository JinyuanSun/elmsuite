import requests, json
import time
import os
from plmsuite.backend import ProteinLanguageModel, PLMErrors

MAX_TOKENS = 1000


class GinkgoPLM(ProteinLanguageModel):
    def __init__(self, **config):
        self.api_key = config

    def prompt_completions(self, prompt, model, **kwargs):
        # Warning("Ginkgo only supports mask-filling completions")
        if "<mask>" not in prompt:
            raise PLMErrors("Ginkgo only supports mask-filling completions")
        chunks = prompt.split("<mask>")
        input_text = []
        for i in range(len(chunks) - 1):
            input_text.extend(list(chunks[i]))
            input_text.append(f"<mask>")
        input_text.extend(list(chunks[-1]))
        sequence = "".join(input_text[:MAX_TOKENS])
        transforms = [{"config": {}, "type": "FILL_MASK"}]
        form_data = {
            "text": sequence,
            "transforms": json.dumps(transforms),
            "model": model,
        }
        response = requests.post(
            "https://api.ginkgobioworks.ai/v1/transforms/run",
            headers={"x-api-key": os.getenv("GINKGO_API_KEY")},
            data=form_data,
        )
        if response.ok:
            while True:
                response = requests.get(
                    response.json()["result"],
                    headers={"x-api-key": os.getenv("GINKGO_API_KEY")},
                )
                if response.ok:
                    response = response.json()
                    if response["status"] == "COMPLETE":
                        return response["result"]
                else:
                    time.sleep(0.5)

    def embed_sequences(self, sequence, model="ginkgo-aa0-650M"):
        transforms = [{"config": {}, "type": "EMBEDDING"}]
        form_data = {
            "text": sequence[:MAX_TOKENS],
            "transforms": json.dumps(transforms),
            "model": model,
        }

        response = requests.post(
            "https://api.ginkgobioworks.ai/v1/transforms/run",
            headers={"x-api-key": os.getenv("GINKGO_API_KEY")},
            data=form_data,
        )

        if response.ok:
            while True:
                response = requests.get(
                    response.json()["result"],
                    headers={"x-api-key": os.getenv("GINKGO_API_KEY")},
                )
                if response.ok:
                    response = response.json()
                    if response["status"] == "COMPLETE":
                        return response["result"]
                else:
                    time.sleep(0.5)


if __name__ == "__main__":
    ginkgo_plm = GinkgoPLM()
    sequence = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    # breakpoint()
    # data = ginkgo_plm.embed_sequences(sequence)
    # breakpoint()
    data = ginkgo_plm.prompt_completions(
        "MTYKLILNGKTLKGETTTEAVDAATAE<mask>VFKQYANDNGVDGEWTYDDATKTFTVTE",
        "ginkgo-aa0-650M",
    )
    print(data)

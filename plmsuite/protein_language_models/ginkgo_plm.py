import requests, json
import time
import os
from plmsuite.backend import ProteinLanguageModel, PLMErrors
from plmsuite.framework.response import PMLResponse

MAX_TOKENS = 1000


class GinkgoPLM(ProteinLanguageModel):
    def __init__(self, **config):
        self.api_key = config
        self.plm_response = PMLResponse()

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
                        return self._normalize_sequence(response, self.plm_response)
                else:
                    time.sleep(0.5)

    def _normalize_sequence(self, response_result, plm_response=None):
        if plm_response is None:
            plm_response = PMLResponse()
        # breakpoint()
        plm_response.choices[0].sequence.content = response_result["result"][0]["result"]['sequence']
        return plm_response

    def _normalize_embedding(self, response_result, plm_response=None):
        if plm_response is None:
            plm_response = PMLResponse()
        plm_response.choices[0].embedding.content = response_result["result"][0]["result"]['embedding']
        return plm_response

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
                        return self._normalize_embedding(response, self.plm_response)
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

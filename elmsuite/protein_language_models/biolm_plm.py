import requests, json
import time
import os
from elmsuite.backend import ProteinLanguageModel, PLMErrors
from elmsuite.framework.response import PMLResponse

MAX_TOKENS = 1000


class BiolmPLM(ProteinLanguageModel):
    def __init__(self, **config):
        self.api_key = config

    def prompt_completions(self, prompt, model, **kwargs):
        url = f"https://biolm.ai/api/v2/{model}/generate/"
        default_param = {
            "temperature": 1,
            "top_p": 1,
            "num_samples": 1,
            "max_length": 384,
        }
        for key in default_param:
            if key in kwargs:
                default_param[key] = kwargs[key]

        payload = json.dumps({"params": default_param, "items": [{"context": prompt}]})
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Token {}".format(os.environ["BIOLM_API_KEY"]),
        }
        # format_as_curl = f"curl -X POST {url} -H 'Content-Type: application/json' - {payload}"
        # print(payload)
        response = requests.request("POST", url, headers=headers, data=payload)

        return response.json()

    def embed_sequences(self, sequence, model="ginkgo-aa0-650M"):
        pass

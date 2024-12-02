import os
import httpx
from plmsuite.backend import ProteinLanguageModel, PLMErrors
from plmsuite.framework.response import PMLResponse


class TogetherPLM(ProteinLanguageModel):
    """
    Together AI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the Fireworks provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("TOGETHER_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Together API key is missing. Please provide it in the config or set the TOGETHER_API_KEY environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def prompt_completions(self, prompt, model, **kwargs):
        """
        Makes a request to the Fireworks AI chat completions endpoint using httpx.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            # Make the request to Fireworks AI endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise PLMErrors(f"Together AI request failed: {http_err}")
        except Exception as e:
            raise PLMErrors(f"An error occurred: {e}")
        


        # Return the normalized response
        return self._normalize_response(response.json())

    def embed_sequences(self, sequences):
        pass

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        plm_response = PMLResponse()
        plm_response.choices[0].message.content = response_data["choices"][0][
            "message"
        ]["content"]
        return plm_response

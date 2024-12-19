from .utils.esm_extract import ESMSequenceExtractor
import requests, json
import time
import os
from elmsuite.backend import ProteinLanguageModel, PLMErrors
from elmsuite.framework.response import PMLResponse
import torch


class LocalPLM(ProteinLanguageModel):
    def __init__(self, **kwargs):
        self.local_plm = {}
        self.plm_response = PMLResponse()

    def _normalize_embedding(self, response_result, plm_response=None):
        if plm_response is None:
            plm_response = PMLResponse()
        plm_response.choices[0].embedding.content = response_result["_"].numpy().ravel()
        return plm_response

    def prompt_completions(self, model, prompt):
        raise NotImplementedError("Prompt completions not implemented for ESM2PLM")

    def embed_sequences(
        self, sequence, model="esm2_t33_650M_UR50D", device="cuda", **kwargs
    ):
        if model not in self.local_plm:
            self.local_plm[model] = ESMSequenceExtractor(
                model_name=model, device=device, **kwargs
            )
        representations = self.local_plm[model].extract_representations(
            [("_", sequence)], pool_method="mean", max_batch_tokens=4096
        )
        return self._normalize_embedding(representations)


if __name__ == "__main__":
    # Initialize the extractor
    extractor = ESMSequenceExtractor(
        model_name="esm1_t34_670M_UR50S",
    )

    # Prepare data
    sequences = [
        ("short1", "MKTVRQ"),
        ("short2", "KALTARQ"),
        ("medium", "MKTVRQERLKSIVRILERSKEPVSGAQ"),
        ("long", "A" * 2048),
    ]

    # Extract representations using the extractor
    representations = extractor.extract_representations(
        sequences, pool_method="mean", max_batch_tokens=4096
    )

    # Use the same model and alphabet from the extractor for manual extraction
    model = extractor.model
    alphabet = extractor.alphabet
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Ensure the model is in eval mode

    # Prepare data in the same order
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(extractor.device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on the same device)
    with torch.no_grad():
        results = model(
            batch_tokens, repr_layers=[extractor.last_layer], return_contacts=False
        )
    token_representations = results["representations"][extractor.last_layer]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        representation = token_representations[i, 1 : tokens_len - 1].mean(0)
        sequence_representations.append(representation.cpu())

    # Calculate delta using the same device and model instance
    delta = sum(representations["long"] - sequence_representations[-1])

    print(f"Delta: {delta}")

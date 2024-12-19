import torch
import esm
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class SequenceGroup:
    sequences: List[Tuple[str, str]]
    total_length: int


class ESMSequenceExtractor:
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_tokens: int = 4096,
    ):
        """
        Initializes the ESM sequence representation extractor.

        Args:
            model_name: The name of the ESM model.
            device: The device to run the model on.
            max_tokens: The maximum number of tokens.
        """
        self.device = device
        self.max_tokens = max_tokens

        # Load model and alphabet
        self.model, self.alphabet = self._load_model(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Get the batch converter
        self.batch_converter = self.alphabet.get_batch_converter()

        # Get the index of the last layer
        self.last_layer = self.model.num_layers

    def _load_model(self, model_name: str) -> Tuple[torch.nn.Module, esm.data.Alphabet]:
        """Loads the specified ESM model."""
        if hasattr(esm.pretrained, model_name):
            return getattr(esm.pretrained, model_name)()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _pack_sequences(
        self, sequences: List[Tuple[str, str]], max_batch_tokens: Optional[int] = None
    ) -> List[SequenceGroup]:
        """
        Smartly packs sequences to group shorter ones together for efficiency.

        Args:
            sequences: A list of sequences.
            max_batch_tokens: The maximum number of tokens per batch, defaults to max_tokens.

        Returns:
            A list of packed sequence groups.
        """
        if max_batch_tokens is None:
            max_batch_tokens = self.max_tokens

        # Sort sequences by length (from longest to shortest)
        sorted_sequences = sorted(sequences, key=lambda x: len(x[1]), reverse=True)

        sequence_groups = []
        current_group = []
        current_length = 0

        for label, seq in sorted_sequences:
            seq_length = len(seq) + 2  # Add 2 for start and end tokens

            # If the sequence is too long, create a separate group
            if seq_length > max_batch_tokens:
                raise ValueError(
                    f"Sequence '{label}' length ({len(seq)}) exceeds max_tokens-2 ({max_batch_tokens-2})"
                )

            # If adding the current sequence exceeds the max length, create a new group
            if current_length + seq_length > max_batch_tokens:
                if current_group:
                    sequence_groups.append(
                        SequenceGroup(
                            sequences=current_group, total_length=current_length
                        )
                    )
                current_group = [(label, seq)]
                current_length = seq_length
            else:
                current_group.append((label, seq))
                current_length += seq_length

        # Add the last group
        if current_group:
            sequence_groups.append(
                SequenceGroup(sequences=current_group, total_length=current_length)
            )

        return sequence_groups

    @torch.no_grad()
    def extract_representations(
        self,
        sequences: List[Tuple[str, str]],
        pool_method: str = "mean",
        max_batch_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract sequence representations.

        Args:
            sequences: A list of sequences, each represented as a tuple (label, sequence).
            pool_method: Pooling method, either 'mean' or 'cls'.
            max_batch_tokens: The maximum number of tokens per batch.

        Returns:
            A dictionary containing sequence labels and their corresponding representations.
        """
        if pool_method not in ["mean", "cls"]:
            raise ValueError("pool_method must be either 'mean' or 'cls'")

        # Smartly pack sequences
        sequence_groups = self._pack_sequences(sequences, max_batch_tokens)

        # Store representations for all sequences
        sequence_representations = {}

        for group in sequence_groups:
            # Convert batch
            batch_labels, batch_strs, batch_tokens = self.batch_converter(
                group.sequences
            )
            batch_tokens = batch_tokens.to(self.device)

            # Calculate sequence lengths
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            # Get representations
            results = self.model(
                batch_tokens, repr_layers=[self.last_layer], return_contacts=False
            )
            token_representations = results["representations"][self.last_layer]

            # Generate representations for each sequence
            for i, (label, tokens_len) in enumerate(zip(batch_labels, batch_lens)):
                if pool_method == "mean":
                    # Use mean pooling (excluding start and end tokens)
                    representation = token_representations[i, 1 : tokens_len - 1].mean(
                        0
                    )
                else:  # cls
                    # Use the representation of the first token as the sequence representation
                    representation = token_representations[i, 0]

                sequence_representations[label] = representation.cpu()

        return sequence_representations

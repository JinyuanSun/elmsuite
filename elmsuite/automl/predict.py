import pickle
import numpy as np
from typing import Union, Dict, List
from .train import TrainingPipeline


class PredictionPipeline:
    def __init__(self, source: Union[str, "TrainingPipeline"], interface=None):
        """Initialize PredictionPipeline from either a trained TrainingPipeline or a path to saved models

        Args:
            source: Either a TrainingPipeline object or path to the pickle file containing trained models
            interface: The interface object for generating embeddings
        """
        self.interface = interface

        if isinstance(source, str):
            # Load from pickle file
            with open(source, "rb") as f:
                self.trained_models = pickle.load(f)
        else:
            # Load from TrainingPipeline object
            self.trained_models = source.trained_models

        # Get feature extraction model name
        self.feature_from = self.trained_models["feature_from"]

        # Group models by prefix
        self.model_groups = {}
        for model_name in self.trained_models.keys():
            if model_name == "feature_from":
                continue
            prefix = model_name.split(".")[0]
            if prefix not in self.model_groups:
                self.model_groups[prefix] = []
            self.model_groups[prefix].append(model_name)

    def _get_embedding(self, sequence: str) -> np.ndarray:
        """Get embedding for a sequence using the pretrained model"""
        results = self.interface.infer.embed.create(
            model=self.feature_from, sequence=sequence
        )
        embedding = np.array(results.choices[0].embedding.content)
        return embedding

    def predict(
        self, sequence: str, model_prefix: str = None
    ) -> Dict[str, List[float]]:
        """Make predictions for a sequence using specified model prefix

        Args:
            sequence: Input protein sequence
            model_prefix: Model type to use (e.g. 'RandomForest'). If None, use all models.

        Returns:
            Dictionary with model prefixes as keys and lists of predictions from different folds as values
        """
        # Get embedding for the sequence
        embedding = self._get_embedding(sequence)

        # Prepare input for prediction
        X = embedding.reshape(1, -1)

        predictions = {}

        # If no specific model prefix is provided, use all models
        prefixes_to_use = [model_prefix] if model_prefix else self.model_groups.keys()

        for prefix in prefixes_to_use:
            if prefix not in self.model_groups:
                raise ValueError(f"Model prefix '{prefix}' not found in trained models")

            fold_predictions = []
            for model_name in self.model_groups[prefix]:
                model = self.trained_models[model_name]
                pred = model.predict(X)[0]
                fold_predictions.append(float(pred))

            predictions[prefix] = fold_predictions

        return predictions

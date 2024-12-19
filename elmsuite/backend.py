from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools


class PLMErrors(Exception):
    """Custom exception for PLM errors."""

    def __init__(self, message):
        super().__init__(message)


class ProteinLanguageModel(ABC):
    @abstractmethod
    def prompt_completions(self, model, prompt):
        """Abstract method for prompt completion calls, to be implemented by each plm."""
        pass

    @abstractmethod
    def embed_sequences(self, sequences):
        """Abstract method for embedding sequences, to be implemented by each plm."""
        pass


class PLMFactory:
    """Factory to dynamically load plm instances based on naming conventions."""

    PLM_DIR = Path(__file__).parent / "protein_language_models"

    @classmethod
    def create_plm(cls, plm_key, config):
        """Dynamically load and create an instance of a plm based on the naming convention."""
        # Convert plm_key to the expected module and class names
        plm_class_name = f"{plm_key.capitalize()}PLM"
        plm_module_name = f"{plm_key}_plm"

        module_path = f"elmsuite.protein_language_models.{plm_module_name}"

        # Lazily load the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module {module_path}: {str(e)}. Please ensure the plm is supported by doing PLMFactory.get_supported_plms()"
            )

        # Instantiate the plm class
        plm_class = getattr(module, plm_class_name)
        return plm_class(**config)

    @classmethod
    @functools.cache
    def get_supported_plm(cls):
        """List all supported plm names based on files present in the plms directory."""
        plm_files = Path(cls.PLM_DIR).glob("*_plm.py")
        return {file.stem.replace("_plm", "") for file in plm_files}

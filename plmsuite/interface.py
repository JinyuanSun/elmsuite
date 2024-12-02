from .backend import PLMFactory

class Interface:
    def __init__(self, plm_configs: dict = {}):
        """
        Initialize the interface with protein language model backends.
        Use the PLMFactory to create protein language model instances.

        Args:
            plm_configs (dict): A dictionary containing plm configurations.
                Each key should be a plm string (e.g., "ginkgo" or "esm"),
                and the value should be a dictionary of configuration options for that provider.
                For example:
                {
                    "ginkgo": {"api_key": "your_ginkgo_api_key"},
                }
        """
        self.plms = {}
        self.plm_configs = plm_configs
        self._infer = None
        self._initialize_plms()

    def _initialize_plms(self):
        """Helper method to initialize or update plms."""
        for plm_key, config in self.plm_configs.items():
            plm_key = self._validate_plm_key(plm_key)
            self.plms[plm_key] = PLMFactory.create_plm(
                plm_key, config
            )

    def _validate_plm_key(self, plm_key):
        """
        Validate if the plm key corresponds to a supported plm.
        """
        supported_plms = PLMFactory.get_supported_plm()

        if plm_key not in supported_plms:
            raise ValueError(
                f"Invalid plm key '{plm_key}'. Supported plms: {supported_plms}. "
                "Make sure the model string is formatted correctly as 'plm:model'."
            )

        return plm_key
    
    @property
    def infer(self):
        """Return the infer API interface."""
        if not self._infer:
            self._infer = Infer(self)
        return self._infer
    # def configure(self, plm_configs: dict = None):

class Infer:
    def __init__(self, interface: "Interface"):
        self.interface = interface
        self._completion = Completions(self.interface)
        self._embed = Embeddings(self.interface)

    @property
    def completion(self):
        """Return the completion API interface."""
        return self._completion
    
    @property
    def embed(self):
        """Return the embed API interface."""
        return self._embed
    
class Completions:
    def __init__(self, interface: "Interface"):
        self.interface = interface

    def create(self, model, prompt, **kwargs):
        """
        Embed sequences using a protein language model.

        Args:
            model (str): The model name.
            sequence (str): The sequence to embed.

        Returns:
            dict: The embedding response.
        """
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'plm:model', got '{model}'"
            )
        plm_key, model_name = model.split(":", 1)
        supported_plms = PLMFactory.get_supported_plm()
        if plm_key not in supported_plms:
            raise ValueError(
                f"Invalid plm '{plm_key}'. Supported plms: {supported_plms}. "
                "Make sure the model string is formatted correctly as 'plm:model'."
            )
        
        if plm_key not in self.interface.plms:
            config = self.interface.plm_configs.get(plm_key, {})
            self.interface.plms[plm_key] = PLMFactory.create_plm(plm_key, config)


        plm = self.interface.plms.get(plm_key)
        if not plm:
            raise ValueError(f"PLM '{plm}' not found.")
        return plm.prompt_completions(prompt, model_name, **kwargs)
    

class Embeddings:
    def __init__(self, interface: "Interface"):
        self.interface = interface

    def create(self, model, sequence, **kwargs):
        """
        Embed sequences using a protein language model.

        Args:
            model (str): The model name.
            sequence (str): The sequence to embed.

        Returns:
            dict: The embedding response.
        """
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'plm:model', got '{model}'"
            )
        plm_key, model_name = model.split(":", 1)
        supported_plms = PLMFactory.get_supported_plm()
        if plm_key not in supported_plms:
            raise ValueError(
                f"Invalid plm '{plm_key}'. Supported plms: {supported_plms}. "
                "Make sure the model string is formatted correctly as 'plm:model'."
            )
        
        if plm_key not in self.interface.plms:
            config = self.interface.plm_configs.get(plm_key, {})
            self.interface.plms[plm_key] = PLMFactory.create_plm(plm_key, config)


        plm = self.interface.plms.get(plm_key)
        if not plm:
            raise ValueError(f"PLM '{plm}' not found.")
        return plm.embed_sequences(sequence, model_name, **kwargs)


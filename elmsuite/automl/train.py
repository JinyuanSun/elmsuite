import pandas as pd
from .data import DataPipeline
import numpy as np
from .utils import _init_mlmodels, _train_model
from scipy.stats import pearsonr, spearmanr
import pickle


class TrainingPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        wt_seq: str,
        pretrained_model: str,
        identifier_col: str,
        output_dir: str,
        interface,
        test_size: float = None,
        random_state: int = 42,
        cv_folds: int = 5,
        models: list = [
            "RandomForest",
            "ExtraTrees",
            "BayesianRidge",
            "GaussianProcessRegressor",
        ],
    ):
        self.df = df
        self.target_col = target_col
        self.identifier_col = identifier_col
        self.pretrained_model = pretrained_model
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.wt_seq = wt_seq
        # if self.cv_folds != -1 and self.test_size is not None:
        #     Warning("Cross-validation is turned on, test_size will be ignored")
        self.output_dir = output_dir
        self.interface = interface
        self.models = models
        self.models_dict = _init_mlmodels(self.models)
        self.mertics = []
        self.trained_models = {
            "feature_from": self.pretrained_model,
        }

    def prepare_data(self):
        self.datapipe = DataPipeline(
            df=self.df,
            target_col=self.target_col,
            wt_seq=self.wt_seq,
            pretrained_model=self.pretrained_model,
            identifier_col=self.identifier_col,
            output_dir=self.output_dir,
            test_size=self.test_size,
            random_state=self.random_state,
            cv_folds=self.cv_folds,
        )

        self.datapipe.preprocess()
        self.datapipe.split_data()
        self.datapipe.retrieve_embeddings(self.interface)
        self.data_info = self.datapipe.generate_data_info_dict()
        self.feat_dir = self.data_info["embedding_dir"]
        self.feat_dict = {}
        for name in self.df[self.identifier_col].values:
            self.feat_dict[name] = np.load(f"{self.feat_dir}/{name}.npy")

    def cross_validation(self):
        train_type, size = self.data_info["train_type"].split(":")
        assert train_type == "cross_validation", "Cross-validation is not turned on"
        assert size == str(self.cv_folds), "Number of folds does not match"
        assert int(size) > 2, "Number of folds must be greater than 2"

        for i in range(self.cv_folds):
            self.models_dict = _init_mlmodels(self.models)
            train = pd.read_csv(f"{self.output_dir}/train_{i}.csv")
            test = pd.read_csv(f"{self.output_dir}/test_{i}.csv")
            X_train = np.array(
                [self.feat_dict[name] for name in train[self.identifier_col].values]
            )
            y_train = train[self.target_col].values
            X_test = np.array(
                [self.feat_dict[name] for name in test[self.identifier_col].values]
            )
            y_test = test[self.target_col].values
            for model_name, model in self.models_dict.items():
                # breakpoint()
                trained_model, y_pred = _train_model(
                    model, X_train, y_train, X_test, y_test
                )
                mse_loss = np.mean((y_pred - y_test) ** 2)
                pcc = pearsonr(y_pred, y_test)[0]
                spr = spearmanr(y_pred, y_test)[0]
                print(
                    f"Model: {model_name:<15} | Fold: {i:>2} | MSE: {mse_loss:.4e} | PCC: {pcc:.4f} | SPR: {spr:.4f}"
                )
                self.mertics.append(
                    {
                        "model": model_name,
                        "fold": i,
                        "mse": mse_loss,
                        "pcc": pcc,
                        "spr": spr,
                    }
                )
                self.trained_models[f"{model_name}.{i}"] = trained_model

        self.df_metrics = pd.DataFrame(self.mertics)
        self.df_metrics.to_csv(f"{self.output_dir}/metrics.csv", index=False)
        with open(f"{self.output_dir}/trained_models.pkl", "wb") as f:
            pickle.dump(self.trained_models, f)

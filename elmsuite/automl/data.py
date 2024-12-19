from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import os
from os.path import join as opj
from tqdm import tqdm


def _make_mutant(wt_seq: str, mutations: str):
    """
    Args:
        wt_seq: wild-type sequence
        mutations: list of mutations in the format "A100C,A101C"
    """
    wt_seq = list(wt_seq)
    for mutation in mutations.split(","):
        aa = mutation[0]
        pos = int(mutation[1:-1]) - 1
        assert (
            wt_seq[pos] == aa
        ), f"Mutation {mutation} does not match wild-type sequence"
        wt_seq[pos] = mutations[-1]
    return "".join(wt_seq)


class DataPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        wt_seq: str,
        pretrained_model: str,
        identifier_col: str,
        output_dir: str,
        test_size: float = None,
        random_state: int = 42,
        cv_folds: int = -1,
    ):
        self.df = df
        self.target_col = target_col
        self.identifier_col = identifier_col
        self.pretrained_model = pretrained_model
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.wt_seq = wt_seq
        if self.cv_folds != -1 and self.test_size is not None:
            Warning("Cross-validation is turned on, test_size will be ignored")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess(self):
        print(f"Original data shape: {self.df.shape}")
        self.df = self.df.dropna()
        self.df = self.df[[self.identifier_col, self.target_col]]
        self.df = self.df.groupby(self.identifier_col).mean().reset_index()
        print(f"Preprocessed data shape: {self.df.shape}")
        self.df["sequence"] = self.df[self.identifier_col].apply(
            lambda x: _make_mutant(self.wt_seq, x)
        )
        # breakpoint()
        self.df.to_csv(f"{self.output_dir}/preprocessed_data.csv", index=False)

    def split_data(self):
        if self.cv_folds == -1:
            train, test = train_test_split(
                self.df, test_size=self.test_size, random_state=self.random_state
            )
            train.to_csv(f"{self.output_dir}/train.csv", index=False)
            test.to_csv(f"{self.output_dir}/test.csv", index=False)
        else:
            kfold = KFold(
                n_splits=self.cv_folds, random_state=self.random_state, shuffle=True
            )
            for i, (train_idx, test_idx) in enumerate(kfold.split(self.df)):
                train = self.df.iloc[train_idx]
                test = self.df.iloc[test_idx]
                train.to_csv(f"{self.output_dir}/train_{i}.csv", index=False)
                test.to_csv(f"{self.output_dir}/test_{i}.csv", index=False)

    def retrieve_embeddings(self, interface):
        model = self.pretrained_model.split(":")[1]
        output_dir = opj(self.output_dir, "embeddings", model)
        os.makedirs(output_dir, exist_ok=True)
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            sequence = row["sequence"]
            identifier = row[self.identifier_col]
            if os.path.exists(opj(output_dir, f"{identifier}.npy")):
                continue
            else:
                results = interface.infer.embed.create(
                    model=self.pretrained_model, sequence=sequence
                )
                embedding = np.array(results.choices[0].embedding.content)
                np.save(opj(output_dir, f"{identifier}.npy"), embedding)

    def generate_data_info_dict(self):
        data_info = {
            "wt_seq": self.wt_seq,
            "pretrained_model": self.pretrained_model,
            "target_col": self.target_col,
            "identifier_col": self.identifier_col,
            "output_dir": self.output_dir,
            "train_type": f"train_test_split:{self.test_size}"
            if self.cv_folds == -1
            else f"cross_validation:{self.cv_folds}",
            "embedding_dir": opj(
                self.output_dir, "embeddings", self.pretrained_model.split(":", 1)[1]
            ),
        }
        return data_info

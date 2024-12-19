import pandas as pd
from .train import TrainingPipeline
from .predict import PredictionPipeline
import numpy as np
import elmsuite as elm
import os


def auto_train(csv_file, wt_seq, pretrained_model, **kwargs):
    df = pd.read_csv(csv_file)
    csv_file_location = os.path.abspath(csv_file)
    output_dir = os.path.dirname(csv_file_location)
    target_col = "label"
    identifier_col = "mutation"
    interface = elm.Interface()
    model_name = pretrained_model.split(":")[1]
    training = TrainingPipeline(
        df=df,
        target_col=target_col,
        wt_seq=wt_seq,
        pretrained_model=pretrained_model,
        identifier_col=identifier_col,
        output_dir=f"{output_dir}/{model_name}-output",
        interface=interface,
        **kwargs,
    )
    training.prepare_data()
    training.cross_validation()
    predictor = PredictionPipeline(training, interface)
    return training, predictor

def _init_mlmodels(models):
    mlmodels = {}
    if "RandomForest" in models:
        from sklearn.ensemble import RandomForestRegressor

        mlmodels["RandomForest"] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    if "ExtraTrees" in models:
        from sklearn.ensemble import ExtraTreesRegressor

        mlmodels["ExtraTrees"] = ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    if "BayesianRidge" in models:
        from sklearn.linear_model import BayesianRidge

        mlmodels["BayesianRidge"] = BayesianRidge()
    if "GaussianProcessRegressor" in models:
        from sklearn.gaussian_process import GaussianProcessRegressor

        mlmodels["GaussianProcessRegressor"] = GaussianProcessRegressor()
    return mlmodels


def _train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

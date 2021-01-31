"""
A script to forecast PV generation for the PoD Challenge.

- First Authored: 2021-01-30
- Owen Huxley <othuxley1@sheffield.ac.uk>
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting
import sklearn.gaussian_process as gp
from datetime import timedelta


class DataProcessing:

    def __init__(self, weather_file, PV_file, capacity):
        self.weather_data = pd.read_csv(weather_file)
        self.PV_data = pd.read_csv(PV_file)
        self.capacity = capacity # MW
        self.PV_train = None
        self.PV_validation = None
        self.weather_train = None
        self.weather_validation = None
        self.munging()
        self.train_validation_split()

    def munging(self):
        # PV data
        self.PV_data["generation_MWh"] = self.PV_data["pv_power_mw"] * 2
        self.PV_data["yield"] = self.PV_data["pv_power_mw"] / self.capacity
        self.PV_data.dropna(inplace=True)
        self.PV_data["datetime"] = pd.to_datetime(self.PV_data.datetime)
        self.PV_data.set_index("datetime", inplace=True)
        self.PV_data["hour"] = self.PV_data.index.hour

        # weather data
        self.weather_data["datetime"] = pd.to_datetime(self.weather_data.datetime)
        self.weather_data.set_index("datetime", inplace=True)
        self.weather_data = self.weather_data.resample("30T", label="left").bfill()
        self.weather_data.iloc[:, -6:] *= 0.5

    def train_validation_split(self):
        mask = self.PV_data.index < "2018-07-08"
        train_index = self.PV_data.loc[mask].index
        validation_index = self.PV_data[~mask].index

        self.PV_train = self.PV_data.loc[train_index].copy()
        self.PV_validation = self.PV_data.loc[validation_index].copy()

        self.weather_train = self.weather_data.loc[train_index].copy()
        self.weather_validation = self.weather_data.loc[validation_index].copy()


class LocalWeatherForecast(DataProcessing):

    def __init__(self, weather_file, PV_file, capacity, test, test_length):
        DataProcessing.__init__(self, weather_file, PV_file, capacity)
        self.testing = test
        self.test_length = test_length


    def features_selection(self):

        X = self.weather_train
        y = self.PV_train["irradiance_Wm-2"]
        # import pdb; pdb.set_trace()
        # bestfeatures = SelectKBest(score_func=f_regression, k=5)
        # fit = bestfeatures.fit(X, y)
        # dfscores = pd.DataFrame(fit.scores_)
        # dfcolumns = pd.DataFrame(X.columns)
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(X)
        X_new.shape
        return dfscores, dfcolumns

    def PCA(self):
        X = self.weather_train.values
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        pca = PCA(n_components = 0.99)
        X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
        print(pca.n_components_)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        # import pdb; pdb.set_trace()
        pd.DataFrame(pca.components_, columns = self.weather_data.columns)
        n_pcs= pca.n_components_ # get number of component
        # get the index of the most important feature on EACH component
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        initial_feature_names = self.weather_data.columns
        # get the most important feature names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

        return list(set(most_important_names))

    def predictions(self, weather_parameter):
        if self.testing:
            start = self.PV_train.index.max() - timedelta(weeks=self.test_length)
            PV_train_4week_subset = self.PV_train.loc[self.PV_train.index > start]
            weather_train_4week_subset = self.weather_train.loc[self.weather_train.index > start]
            X_tr = weather_train_4week_subset[[*self.PCA()]]
            y_tr = PV_train_4week_subset[[weather_parameter]]
            X_te = self.weather_validation[[*self.PCA()]]
            y_te = self.PV_validation[[weather_parameter]]
        else:
            X_tr = self.weather_train[[*self.PCA()]]
            y_tr = self.PV_train[[weather_parameter]]
            X_te = self.weather_validation[[*self.PCA()]]
            y_te = self.PV_validation[[weather_parameter]]

        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) \
                 * gp.kernels.RBF(10.0, (1e-3, 1e3))

        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            n_restarts_optimizer=10,
                                            alpha=0.1, normalize_y=True)
        model.fit(X_tr, y_tr)
        params = model.kernel_.get_params()
        y_pred, std = model.predict(X_te, return_std=True)
        MSE = ((y_pred-y_te)**2).mean()

        results = pd.DataFrame(data=np.concatenate((y_pred, y_te), axis=1), index=y_te.index, columns=["Predicitons", "Actuals"])
        results.plot()
        return y_pred

    def test_model(self):
        return

    def predict(self):
        return


class PVForecast(DataProcessing):

    def __init__(self, weather_file, PV_file, capacity, test, test_length):
        DataProcessing.__init__(self, weather_file, PV_file, capacity)
        self.testing = test
        self.test_length = test_length

    def train_model(self, irradiance_test, panel_temp_test):
        if self.testing:
            start = self.PV_train.index.max() - timedelta(weeks=self.test_length)
            PV_train_4week_subset = self.PV_train.loc[self.PV_train.index > start]
            X_tr = PV_train_4week_subset[["irradiance_Wm-2", "panel_temp_C", "hour"]]
            y_tr = PV_train_4week_subset[["yield"]]
            X_te = self.PV_validation[["irradiance_Wm-2", "panel_temp_C", "hour"]]
            X_te.loc[:, "irradiance_Wm-2"] = irradiance_test
            X_te.loc[:, "panel_temp_C"] = panel_temp_test
            y_te = self.PV_validation[["yield"]]
        else:
            X_tr = self.PV_train[["irradiance_Wm-2", "panel_temp_C", "hour"]]
            y_tr = self.PV_train[["yield"]]
            X_te = self.PV_validation[["irradiance_Wm-2", "panel_temp_C", "hour"]]
            y_te = self.PV_validation[["yield"]]



        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) \
                 * gp.kernels.RBF(10.0, (1e-3, 1e3))

        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            n_restarts_optimizer=10,
                                            alpha=0.1, normalize_y=True)
        model.fit(X_tr, y_tr)
        params = model.kernel_.get_params()
        # import pdb; pdb.set_trace()
        y_pred, std = model.predict(X_te, return_std=True)
        MSE = ((y_pred-y_te)**2).mean()

        results = pd.DataFrame(data=np.concatenate((y_pred, y_te), axis=1), index=y_te.index, columns=["Predicitons", "Actuals"])
        results.plot()
        results.to_csv("GaussianProcess_PV_Estimates.csv")




if __name__ == "__main__":
    w_file = "../data/weather_train_set0.csv"
    PV_file = "../data/pv_train_set0.csv"
    self = LocalWeatherForecast(w_file, PV_file, 5, True, 1)
    irr_predictions = self.predictions("irradiance_Wm-2")
    panel_temp_predictions = self.predictions("panel_temp_C")
    self = PVForecast(w_file, PV_file, 5, True, 1)
    self.train_model(irr_predictions, panel_temp_predictions)

    import pdb; pdb.set_trace()

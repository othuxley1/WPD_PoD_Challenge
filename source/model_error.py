
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
import matplotlib.pyplot as plt
import calmap

class testing:

    def __init__(self, model_output):
        self.model = pd.read_csv(model_output)
        self.y_predict = self.model["Predicitons"]
        self.y_true = self.model["Actuals"]
        self.model_yield = self.model["Predicitons"] - self.model["Actuals"]
        #print(np.median(self.y_true))

    def heatmap(self):
        days_indx = [0]
        ntime_step = 1
        ndays = 1
        date0 = self.model['datetime'][0].split(' ')[0]
        time0 = self.model['datetime'][0].split(' ')[1]
        time_ticks = [time0.split(':')[0]]
        days_ticks = [date0.split('-')[2]]
        for i in range(len(self.model['datetime'])-1):
            date = self.model['datetime'][i+1].split(' ')[0]
            time = self.model['datetime'][i+1].split(' ')[1]
            if date == date0:
                ntime_step+=1
                if time.split(':')[1]=='00':
                    time_ticks.append(time.split(':')[0])
            else:
                date0 = date
                ntime_step = 1
                ndays+=1
                days_indx.append(i+1)
                time0 = time
                time_ticks = [time0.split(':')[0]]
                days_ticks.append(date.split('-')[2])

        # print(ntime_step)
        # print(ndays)
        # print(days_indx)
        days_indx.append(len(self.model_yield))


        map = np.zeros((ndays, ntime_step))
        for i in range(ndays):
            map[i] = self.model_yield[days_indx[i]: days_indx[i+1]]

        ax = plt.gca()

        im = ax.imshow(map, cmap='RdBu', vmin=-np.max(abs(map)), vmax=np.max(abs(map)))
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, label='Predicted - true')

        ax.set_xticks(np.arange(0, map.shape[1],2))
        ax.set_xticklabels(time_ticks)
        ax.set_yticks(np.arange(0, ndays))
        ax.set_yticklabels(days_ticks)
        ax.set_xlabel('Time of day')
        ax.set_ylabel('Day of month')

        #plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        plt.show()


if __name__ == "__main__":
    model_output = "GaussianProcess_PV_Estimates.csv"
    self = testing(model_output)
    self.heatmap()

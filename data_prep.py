from operator import index
import numpy as np
import pandas as pd
from itertools import chain
from pandas.core.frame import DataFrame

#from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt

class DataPrep(object):
    """
    This class is used to read CMAPSS data from the provided files
    Inputs:
        - num_settings: Number of operational settings
        - num_sensors: Number of sensor measurements
        - num_units: Number of units to be used for a particular activity (here for VAE training)
    
    Output:
        - df: Readied dataframe
    """
    def __init__(self,
                 file=None,
                 step=None,
                 num_settings=3,
                 num_sensors=21,
                 num_units=100,
                 normalization_type="01") -> None:
        super().__init__()
        self.file = file
        self.step = step
        self.num_settings = num_settings
        self.num_sensors = num_sensors
        self.num_units = num_units

        self.normalization_type = normalization_type

    def ReadData(self) -> DataFrame:
        df = pd.read_table(self.file, header=None, sep="\s+")
        column_names = self._ColNames()
        df.columns = column_names
        if self.step == "VAE":
            df = df[df[column_names[0]] <= self.num_units]
        elif self.step == "RL":
            df = df[(df[column_names[0]] > self.num_units) & (df[column_names[0]] <= 2*self.num_units)]
            df = df.reset_index(drop=True) # Necessary for setting index back to 0
        else:
            df = df[df[column_names[0]] > 2*self.num_units]
            df = df.reset_index(drop=True) # Necessary for setting index back to 0
        RunTimes = self._UnitRunTime(df, column_names)
        self._FeatureSelection(df)
        normalized_values = self._FeatureStandardize(df)
        normalized_time = self._TimeNormalize(df, RunTimes)
        new_df = self._FinalDF(df, normalized_time, normalized_values)

        return new_df

    def _ColNames(self) -> list:
        self.operational_settings = ['OpSetting' + str(i) for i in range(1,self.num_settings+1)]
        self.sensors = ['Sensor' + str(i) for i in range(1,self.num_sensors+1)]
        self.setting_measurement_names = list(chain(*[self.operational_settings, self.sensors]))
        column_names = ['Unit', 'Time'] + self.setting_measurement_names
        
        return column_names
        
    def _UnitRunTime(self,df, column_names) -> list:
        run_times = df.groupby([column_names[0]]).count()[column_names[1]]

        return run_times

    def _FeatureSelection(self, df) -> None:
        nunique = df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        df.drop(cols_to_drop, axis=1)

    def _FeatureStandardize(self, df) -> DataFrame:
        if self.normalization_type == "01":
            normalized_values = df[self.setting_measurement_names].apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)), axis=0)
        else:
            normalized_values = df[self.setting_measurement_names].apply(lambda x: (x - np.mean(x))/np.std(x), axis=0)

        return normalized_values

    def _TimeNormalize(self, df, run_times) -> DataFrame:
        normalized_time = []

        if self.step == "VAE":
            units_cntr = 0
        elif self.step == "RL":
            units_cntr = self.num_units
        else:
            units_cntr = 2*self.num_units

        cntr = 0

        for i in range(1,len(run_times)+1):
            chunk = list(df['Time'][cntr:cntr+run_times[units_cntr+i]]/run_times[units_cntr+i])
            normalized_time.append(chunk[len(chunk)-1::-1])
            cntr += run_times[units_cntr+i]
        normalized_time = list(chain(*normalized_time))
        normalized_time = pd.DataFrame(normalized_time, columns=['NormTime'])

        return normalized_time

    def _FinalDF(self, df, normalized_time, normalized_values) -> DataFrame:
        new_df = pd.concat([df['Unit'], normalized_time, normalized_values], axis=1)

        return new_df

class Vec2Img(object):
    """
    Class for transforming time series vector data to images.
    We use Markov Transition Field process
    
    Inputs:
        - df: Dataframe provided in this case from class DataPrep
        - op_image_size: Desired dimensionality for operational settings
        - sens_image_size: Desired dimensionality for sensor measurements

    Outputs:
        - self.im_operational_settings, self.im_sensor_measurements: Numpy arrays representing imaged time series data
    """
    def __init__(self,
                 df=None,
                 data=None,
                 image_size=24,
                 plot=False) -> None:
        super().__init__()
        self.df = df
        self.data = data
        self.image_size = image_size
        self.plot = plot
    
    def Transform(self) -> np.ndarray:
        mtf = MarkovTransitionField(image_size=self.image_size)
        self.images = mtf.transform(self.df[self.data.setting_measurement_names])

        return self.images

    def PlotImg(self) -> None:
        if self.plot:
            plt.imshow(self.images[0])
            plt.show()


if __name__ == "__main__":
    file_path = "CMAPSSData/train_FD002.txt"
    num_settings = 3
    num_sensors = 21
    num_units = 100
    step = "VAE"

    data = DataPrep(file=file_path,
                    num_settings=num_settings, 
                    num_sensors=num_sensors, 
                    num_units=num_units, 
                    step=step,
                    normalization_type="01")
    
    df = data.ReadData()
    print(df.columns)
    
    image_data = Vec2Img(df=df,
                         data=data,
                         image_size=num_settings+num_sensors,
                         plot=True)
    
    images = image_data.Transform()
    image_data.PlotImg()
    
    
    

    
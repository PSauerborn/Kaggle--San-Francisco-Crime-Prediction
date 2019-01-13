
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class dataTool():
    """Convenience tool used to analyse data sets"""

    def fit(self, data):
        """Function used to fit tool to a specific data set

        Parameters
        ----------
        data: pandas DataFrame
            data to be analyzed. Note that this must be passed down as a pandas DataFrame object
        """

        self.data = data

        self.dimensions = data.shape

        self.find_nan(self.data)

        self.find_numeric()

    def find_nan(self, data, plot=False):
        """Function used to find the NaN values within a dataset

        Parameters
        ----------
        data: pandas DataFrame object
            data to be analysed
        plot: Boolean (default=False)
            plots the ratio of nan values if set to True

        Attributes
        ----------
        self.data_nan: pandas series object
            series object containing the total number of nan values present in each column
        self.nan_columns: list
            list containing the names of columns with nan values
        self.nan_ratios: list
            list containing tuples with the form (column_name, ratio) where ratio indicates what percentage of the column consists of nan values

        """

        self.data_nan = np.sum(data.isna())
        self.data_nan = self.data_nan[self.data_nan != 0]

        self.nan_columns = [i for i in self.data_nan.index]

        self.nan_ratios = [(i, a / self.dimensions[0])
                           for (i, a) in zip(self.data_nan.index, self.data_nan.values)]

        if plot:
            self.plot_nan()

    def plot_nan(self):
        """Function used to plot the nan_rations list"""

        try:

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.3)

            ax.bar(np.arange(0, self.data_nan.shape[0]),
                   self.data_nan.values / self.dimensions[0], width=0.4)
            ax.set_xticks(np.arange(0, self.data_nan.shape[0]))

            ax.set_xticklabels(self.data_nan.index, rotation='vertical')
            ax.set_ylabel('Ratio of NaN Values')
            plt.show()

        except:
            print('Error: No Fitted Data')

    def find_numeric(self, plot=False):
        """Function used to evaluate which columns contain numeric data and which columns contain non-numeric data

        Parameters
        ----------
        data: pandas DataFrame object
            data to be analyzed
        plot: Boolean (default=False)
            is set to True, the number of categorical and numerical columns is plotted

        Attributes
        ----------
        self.numerical: list
            list containing columns names of columns with numerical data
        self.categorical: list
            list containing column names of columns with non-nummerical cata

        """

        numeric_types = ['float64', 'float32', 'float16', 'float_', 'int_', 'int8',
                         'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']

        self.numerical = []
        self.categorical = []

        for column in self.data.columns:
            if self.data[column].dtype in numeric_types:
                self.numerical.append(column)
            else:
                self.categorical.append(column)


        if plot:

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)
            ax.bar(np.arange(0, 2), [len(self.numerical), len(self.categorical)], width=0.4)
            ax.set_xticks(np.arange(0, 2))
            ax.set_xticklabels(['Numeric', 'Categorical'], rotation='vertical')
            ax.set_ylabel('Total Entries')
            plt.show()

    def sort_data(self, reverse=False):

        self.data = self.data[self.numerical + self.categorical]

        return self.data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from dataTool import dataTool
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer


def process_data(data, return_data=False, save=True):

    # first the target and data are seperated

    target, data = data['Category'], data.drop(columns=['Category'])

    # the data tool is then instantiated

    tool = dataTool()
    tool.fit(data)

    data = tool.sort_data()

    # the numeric data is standardized

    s = StandardScaler()

    data[tool.numerical] = s.fit_transform(data[tool.numerical])

    # the target data is then onehot encoded

    target = pd.get_dummies(target, drop_first=True)

    # bag, vocab = process_desc(data['Descript'])

    dates = format_dates(data['Dates'])

    data.drop(columns=['Dates', 'Address', 'Descript'], inplace=True)

    data = pd.concat((dates, data), axis=1)

    data = pd.get_dummies(data, columns=['DayOfWeek', 'Resolution', 'PdDistrict', 'year', 'month'], drop_first=True)

    data['dayx'] = np.cos(2*np.pi*data['day'] / 31)
    data['dayy'] = np.sin(2*np.pi*data['day'] / 31)

    data.drop(columns=['day'], inplace=True)


    if save:
        data.to_csv('./data/train_formatted.csv', index=False)
        target.to_csv('./data/train_labels_formatted.csv', index=False)

    if return_data:
        return data, target


def process_desc(data):
    """Function used to convert the 'description' column to a bag of words model"""

    count = CountVectorizer()

    bag = count.fit_transform(data)

    vocab = count.vocabulary_

    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

    data = tfidf.fit_transform(count.fit_transform(docs)).toarray()

    return data, vocab


def format_dates(data):


    dates = np.zeros((data.shape[0], 5))

    for i, entry in enumerate(data.values):

        date, time = entry.split(' ')

        year, month, day = date.split('-')
        hour, minute, second = time.split(':')

        tx, ty = np.cos(2*np.pi*int(hour) / 24), np.sin(2*np.pi*int(hour) / 24)

        dates[i, 0], dates[i, 1], dates[i, 2] = int(year), int(month), int(day)
        dates[i, 3], dates[i, 4] = tx, ty

    dates = pd.DataFrame(dates, columns=['year', 'month', 'day', 'tx', 'ty'])

    return dates

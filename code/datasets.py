from numpy import *
import util
import csv

class PriceDataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    val_x = None # X (data) of validation set.
    val_y = None # Y (label) of validation set.

    def __init__(self):
        ## read the csv for training (price_data_tr.csv), 
        #                   val (price_data_val.csv)
        #                   and testing set (price_data_ts.csv)
        #
        ## CAUTION: the first row is the header 
        ##          (there is an option to skip the header 
        ##            when you read csv in python csv lib.)
        data_tr = open('/Users/sinjisu/ml2019fallca02-jsshin17/dataset/price_data_tr.csv', 'r')
        lines = csv.reader(data_tr)
        i = 0
        x = []
        y = []

        for line in lines:
            if i >= 1:
                temp = []
                temp.extend(line[3:5])
                temp.extend(line[6:])
                x.append(temp)
                y.append(line[2])
            i += 1

        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j]=double(x[i][j])
        for i in range(len(y)):
            y[i] = double(y[i])
        self.tr_x = x
        self.tr_y = reshape(y, (-1, 1))
        data_tr.close()
        x = []
        y = []

        i = 0
        data_val = open('/Users/sinjisu/ml2019fallca02-jsshin17/dataset/price_data_val.csv', 'r')
        lines = csv.reader(data_val)
        for line in lines:
            if i >= 1:
                temp = []
                temp.extend(line[3:5])
                temp.extend(line[6:])
                x.append(temp)
                y.append(line[2])
            i += 1

        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = double(x[i][j])
        for i in range(len(y)):
            y[i] = double(y[i])
        self.val_x = x
        self.val_y = reshape(y, (-1, 1))
        data_val.close()

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.val_x, self.val_y]

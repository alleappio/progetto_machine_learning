from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class PrepareData:
    def __init__(self, data, target = 'critical_temp', normalization=True):
        self.data = data
        self.target = target
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.4)
        
        self.train_data = self.train_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)
        
        self.split_X_y()
        if normalization:
            self.normalization()

    def split_X_y(self):
        self.y_train = self.train_data[self.target]
        self.X_train = self.train_data.drop(self.target, axis=1)
        self.y_test = self.test_data[self.target]
        self.X_test = self.test_data.drop(self.target, axis=1)

    def normalization(self):
        mms = MinMaxScaler()
        
        mms.fit(self.X_train)
        self.X_train[self.X_train.columns] = mms.transform(self.X_train[self.X_train.columns])
        
        mms.transform(self.X_test)
        self.X_test[self.X_test.columns] = mms.transform(self.X_test[self.X_test.columns])
    
    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
        return self.X_test, self.y_test


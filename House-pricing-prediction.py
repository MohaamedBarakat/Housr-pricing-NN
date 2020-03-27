import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
class HousePricing():
    def __init__(self):
        pass
    def filepath(self,file_path):
        df = pd.read_csv(file_path)
        return df
    def dataSet(self,df):
        data_set = df.values
        return data_set
    def featuresAndLabel(self,dataSet):
        X = dataSet[:,0:10]
        Y = dataSet[:,10]
        return X,Y
    def scalingFeatures(self,X):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        return X_scale
    def toTrainTest(self,X,Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        return X_train, X_test, Y_train, Y_test
    def toTestValidation(self,X_test,Y_test):
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
        return X_val, X_test, Y_val, Y_test
    def model(self,X_train, Y_train,X_val, Y_val):
        model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy'])
        hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))
        return hist,model
    def getAccuracy(self,model):
        return model.evaluate(X_test, Y_test)[1]
house = HousePricing()
df = house.filepath('housepricedata.csv')
dataSet = house.dataSet(df)
X,Y = house.featuresAndLabel(dataSet)
X = house.scalingFeatures(X)
X_train, X_test, Y_train, Y_test = house.toTrainTest(X,Y)
X_val, X_test, Y_val, Y_test = house.toTestValidation(X_test,Y_test)
hist,model = house.model(X_train, Y_train,X_val, Y_val)
print(house.getAccuracy(model))








#df = pd.read_csv('housepricedata.csv')
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)     

    
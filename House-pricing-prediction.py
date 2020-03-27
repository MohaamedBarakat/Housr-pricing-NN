class HousePricing():
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('housepricedata.csv')
    dataSet = df.values = df.values
    X = dataSet[:,0:10]
    Y = dataSet[:,10]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

    
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import datetime as dt

# First things first 
# I need to format the data so that my machine learning model can predict 
def format_data(csv):
    # Get the data out of the csv file and import it into a Pandas dataframe
    df = pd.read_csv(csv)

    # I need to make x and y data to train the model
    # x data is going to be the open, high, and low prices
    x = df[['Open', 'High', 'Low']]
    # Convert dataframe to numpy array
    x = x.values
    
    # y data is going to be the closing price
    y = df['Close']
    # Convert dataframe to numpy array
    y = y.values

    # Need to separate training data and test data
    x_train = x[:-300]
    x_test = x[-300:]
    y_train = y[:-300]
    y_test = y[-300:]

    # Need to get dates for the last 300 days so that I can plot a timeline
    test_dates = df['Date']
    test_dates = test_dates.values
    test_dates = test_dates[-300:]

    return x_train, y_train, x_test, y_test, test_dates

    
def prediction_model():
    # This scaler will put all of the x data on the same scale.
    # Doing this will yeild better prediction preformance
    scaler = StandardScaler()
    x_train, y_train, x_test, y_test, test_dates = format_data('bitcoin_history.csv')
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # This is the neural net model
    mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(10,2), random_state=1, batch_size='auto')
    
    # This will train the model on the training data
    mlp.fit(x_train, y_train)
    # This gets the predictions given the test data
    predictions = mlp.predict(x_test)
    # This will plot the predicted prices against the actual prices
    dates = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in test_dates]
    
    predction_patch = mpatches.Patch(color='orange', label='Predicted price')
    actual_patch = mpatches.Patch(color='blue', label='Actual price')
    plt.legend(handles=[predction_patch, actual_patch])

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(dates, y_test, dates, predictions)
    plt.show()
    

    print(explained_variance_score(predictions, y_test))

prediction_model()


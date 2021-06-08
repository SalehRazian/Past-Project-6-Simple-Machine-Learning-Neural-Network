import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

#Define the Data

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
#Replace NA data with an outlier to prevent removal of data
df.fillna(-9999, inplace=True)

#How much of the data available woulf be used in the regression
#eg. using data from 10 days ago to predict today
forecast_out = int(math.ceil(0.01*len(df)))


#It will shift the 'adj. close' column up {-forecast_out}
#The Values will be stored in label
#We will try to predict the label with the given info {which is the future}
df['label'] = df[forecast_col].shift(-(forecast_out))


#df.dropna(inplace=True)


X = np.array(df.drop(['label'],1))

#You scale X would normalize the data, it helps with training and testing
#Takes time 
X = preprocessing.scale(X)

X = X[:-forecast_out]
X_lately = X[-forecast_out:] #the really unknown data

#remove the extra bit at the bottom because we want to match X with y
#X = X[:-forcast_out+1] ##We Droped the labeles after the shift so  ////Ignore things changed
df.dropna(inplace=True)
y = np.array(df['label'])

#It shuffle the data and produces the datasets to be used to avoid bias data
#test size the percentage of test size from the whole data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Classifier
#clf = svm.SVR(kernel='linear') #Support Vector Machine
#n_jobs = -1 just uses all the processing power a
clf = LinearRegression(n_jobs=-1) #not needed after save 
clf.fit(X_train, y_train) #not needed after save

#saving the Classifier after training
with open('linearregression.pickle','wb') as f: #not needed after save
    pickle.dump(clf,f) #not needed after save

#Calling and reading the file
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

#testing the file
accuracy = clf.score(X_test, y_test)
print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set)

df['Forecast'] = np.nan # making a colum which is empty next to every value in the table because there is no forecast for them

last_date = df.iloc[-1].name #finding the last date in the table now
last_unix = last_date.timestamp()
one_day = 86400 #seconds
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    #Add in the position of the next day the date, Add NAN to all columns except the last one
    #in the last column add [i] which is the for3casted value

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()










                         
                         




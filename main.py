import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import pickle

def dep_delay(data):
    if data <=0:
        return 0
    elif (data > 0 and data <=5):
          return 1
    elif (data >5 and data <=10):
        return 2
    elif (data > 10 and data <=20):
        return 3
    elif(data > 20 and data <=50):
        return 4
    elif (data >50 and data <=100):
        return 5
    elif (data > 100 and data <= 200):
        return 6
    elif (data > 200 and data <=500):
        return 7
    elif (data > 501 and data <=1000):
        return 8
    else:
        return 9


def scale(data):
    return((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))


data = pd.read_csv('data.csv', index_col=0)
data['DEP_DELAY'] = data['DEPARTURE_DELAY']
data.drop('DEPARTURE_DELAY', axis=1, inplace=True)
data['DEP_DELAY'] = data['DEP_DELAY'].apply(dep_delay)
col_names = list(data.columns)
X= data[col_names[0:23]]
y = data[col_names[23]]
print(X.head())
print(y.head())
data.iloc[:, :-1] = data.iloc[:, :-1].apply(scale)
print(data.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 42)



DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
pred_train = DT.predict(X_train)
print(pd.crosstab(y_train,pred_train))
train_accuracy = np.mean(pred_train == y_train)
print(train_accuracy)
pred_test = DT.predict(X_test)
print(pd.crosstab(y_test,pred_test))
test_accuracy = np.mean(y_test == pred_test)
print(test_accuracy)

import pickle
DT_model = 'DT_model.pkl'
pickle.dump(DT, open(DT_model, 'wb'))



classifier=AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy'),n_estimators=200)
classifier.fit(X_train,y_train)
pred_adaboost_test=classifier.predict(X_test)
test_accuracy_adaboost=np.mean(pred_adaboost_test==y_test)*100
print(pd.crosstab(pred_adaboost_test, y_test))
print(test_accuracy_adaboost)
ada_boost_model = 'adaboost_model.pkl'
pickle.dump(classifier, open(ada_boost_model, 'wb'))

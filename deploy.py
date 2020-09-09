'''
int_features  = ['1',    '3',    '5',    '8',   '12',   '20',   '49',  '324', '2302', '2324',   '34',
          '3',    '2:23',    '2:34',    '1:8',    '3:24',    '1:60', '4:53']

x1 = []
x2 = []
#x3 = [int(x) for x in int_features]
for i in int_features:
    if ':' in i:
        first, second = i.split(':',1)
        x1.append(first)
        x1.append(second)
    else:
        j = int(i)
        x1.append(j)
final_feat = [int(x) for x in x1]

for i in int_features:
    print(i)
    x2.append(i)
    x1.append(i.split(':'))
print(x2)

print(x1)
print(final_feat)
print(len(final_feat))
print(len(x1))
print(type(final_feat[22]))
#print(x3)
#print(list(map(int,x1)))

'''
import pandas as pd
import pickle

def scale(data):
    return((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))

def airline(data):
    if (data == 'American Airlines Inc.' or data == 'AA' ):
        return 0
    elif (data == 'Alaska Airlines Inc.' or data == 'AS'):
        return 1
    elif (data == 'JetBlue Airways' or data == 'B6'):
        return 2
    elif (data == 'Delta Air Lines Inc.' or data == 'DL'):
        return 3
    elif (data == 'Atlantic Southeast Airlines' or data == 'EV'):
        return 4
    elif (data == 'Frontier Airlines Inc.' or data == 'F9'):
        return 5
    elif (data == 'Hawaiian Airlines Inc.'or data == 'HA'):
        return 6
    elif (data == 'American Eagle Airlines Inc.'or data == 'MQ'):
        return 7
    elif (data == 'Spirit Air Lines'or data == 'NK'):
        return 8
    elif (data == 'Skywest Airlines Inc.'or data == 'OO'):
        return 9
    elif (data == 'United Air Lines Inc.' or data == 'UA'):
        return 10
    elif (data == 'US Airways Inc.' or data == 'US'):
        return 11
    elif (data == 'American Eagle Airlines Inc.' or data == 'VX'):
        return 12
    elif (data == 'Southwest Airlines Co.' or data == 'WN'):
        return 13

#model = pickle.load(open('adaboost_model.pkl', 'rb'))

X = pd.read_csv('E:\\pooja\\DS\\Flight Delay Prediction Project\\trial.csv')
print(X)
print(X['AIRLINE'])
X['AIRLINE'] = X['AIRLINE'].apply(airline)
print(X['AIRLINE'])

X[['WHEELS_ON_HOUR', 'WHEELS_ON_MIN']] = X['WHEELS_ON'].str.split(":", expand=True)
X[['WHEELS_OFF_HOUR', 'WHEELS_OFF_MIN']] = X['WHEELS_OFF'].str.split(":", expand=True)
X[['SCHEDULED_DEPARTURE_HOUR', 'SCHEDULED_DEPARTURE_MIN']] = X['SCHEDULED_DEPARTURE'].str.split(":", expand=True)
X[['SCHEDULED_ARRIVAL_HOUR', 'SCHEDULED_ARRIVAL_MIN']] = X['SCHEDULED_ARRIVAL'].str.split(":", expand=True)
X[['DEPARTURE_TIME_HOUR', 'DEPARTURE_TIME_MIN']] = X['DEPARTURE_TIME'].str.split(":", expand=True)
X[['ARRIVAL_TIME_HOUR', 'ARRIVAL_TIME_MIN']] = X['ARRIVAL_TIME'].str.split(":", expand=True)

X.drop(['WHEELS_ON', 'WHEELS_OFF','SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL', 'DEPARTURE_TIME', 'ARRIVAL_TIME'], axis=1, inplace=True)
X = X.astype('int64')
#predict_ans = model.predict(X)
#print(predict_ans)
#X['Result'] = predict_ans
#print(X)
print(X.info())

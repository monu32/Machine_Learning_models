import pandas as pd
import numpy as np

df=pd.read_csv('Train_matches.csv')
pd.set_option("max_columns",None)
df.drop(['city','result','dl_applied','id','venue','player_of_match'],axis=1,inplace=True)
df.drop(df.index[300],axis=0,inplace=True)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df.team1=(lb.fit_transform(df.team1)+1)/10
df.team2=(lb.fit_transform(df.team2)+1)/10
df.winner=(lb.fit_transform(df.winner)+1)/10
df.toss_winner=(lb.fit_transform(df.toss_winner)+1)/10
df.toss_decision=lb.fit_transform(df.toss_decision)+1
df.winner.loc[df.team1==df.winner]=1
df.winner.loc[df.team2==df.winner]=2
df.winner=df.winner.astype(int)

#Apply regression
x=df.drop(['winner'],axis=1)
y=df['winner']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
model=LogisticRegression(fit_intercept=True)
model.fit(x_train,y_train)
predict_value=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(" Accuracy score:",accuracy_score(y_test,predict_value))

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
model=LogisticRegression(fit_intercept=True)
model.fit(x_train,y_train)
predict_value=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(" Accuracy score:",accuracy_score(y_test,predict_value))

# Note: Use "cross_validation" if model_selection give error

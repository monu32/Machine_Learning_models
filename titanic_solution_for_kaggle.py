import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
pd.set_option('max_columns',None)

test_data=pd.read_csv('test.csv')
df.Sex.replace(['male','female'],[0,1],inplace=True)
test_data.Sex.replace(['male','female'],[0,1],inplace=True)
df.drop(['Name','Ticket','SibSp','Parch','Cabin','Embarked'],axis=1,inplace=True)
df.Age.fillna(method='pad',inplace=True)
df.Age=df.Age.astype(int)
test_data.drop(['Name','Ticket','SibSp','Parch','Cabin','Embarked'],axis=1,inplace=True)
test_data.Age.fillna(method='pad',inplace=True)
test_data.Age=df.Age.astype(int)
df.Fare.loc[df.Pclass==1]=df.Fare[df.Pclass==1].mean()
df.Fare.loc[df.Pclass==2]=df.Fare[df.Pclass==2].mean()
df.Fare.loc[df.Pclass==3]=df.Fare[df.Pclass==3].mean()


test_data.Fare.loc[df.Pclass==1]=test_data.Fare[df.Pclass==1].mean()
test_data.Fare.loc[df.Pclass==2]=test_data.Fare[df.Pclass==2].mean()
test_data.Fare.loc[df.Pclass==3]=test_data.Fare[df.Pclass==3].mean()


test_input=test_data[['Fare','Sex','Pclass','Age']]

print(df)
print(test_data)
#Apply regression
x=df[['Fare','Sex','Pclass','Age']]
y=df['Survived']
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)
predict_value=model.predict(test_input)
output=pd.DataFrame(columns=['PassengerId','Survived'])
output['PassengerId']=test_data.PassengerId
output['Survived']=predict_value
output.to_csv('first_file.csv',index=False)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
df=pd.read_csv('spam.csv')
df.result=df.result.map({'ham':0,'spam':1})

from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test=train_test_split(df.text,df.result,random_state=4)
cv=CountVectorizer(stop_words='english')
x_train_cv=cv.fit_transform(x_train)
x_test_cv=cv.transform(x_test)
model=MultinomialNB()
model.fit(x_train_cv,y_train)
predict_value=model.predict(x_test_cv)
print("Accuracy score :",accuracy_score(y_test,predict_value))


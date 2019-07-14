import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

#data preproccesing

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

LabelEncoder_X_1 = LabelEncoder()

X[:,1]=LabelEncoder_X_1.fit_transform(X[:,1])

LabelEncoder_X_2 = LabelEncoder()

X[:,2]=LabelEncoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])

X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#fitting the model

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train,Y_train)


#predict the test

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

c= confusion_matrix(Y_test,y_pred)






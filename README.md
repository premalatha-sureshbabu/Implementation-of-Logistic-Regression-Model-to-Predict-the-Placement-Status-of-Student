# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries which are used for the program.

2.Load the dataset.

3.Check for null data values and duplicate data values in the dataframe.

4.Apply logistic regression and predict the y output.

5.Calculate the confusion,accuracy and classification of the dataset.
 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S.Prema Latha
RegisterNumber:  212222230112

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85,90,80]])
*/
```

## Output:
![Screenshot 2023-08-31 093622](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/c17dd1b4-16d4-442d-b9a5-58f78460e5c3)

![Screenshot 2023-08-31 093631](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/1ecb3e40-6fea-4cb5-a3d5-34208274892e)

![Screenshot 2023-08-31 093640](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/699c6fd8-18ac-4970-bb97-da5ca0e35496)

![Screenshot 2023-08-31 093647](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/06e902a2-06ef-42fe-8476-7cd9e5b73d65)

![Screenshot 2023-08-31 093703](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/7485e36c-1264-44ef-a401-b2f8dfbed022)

![Screenshot 2023-08-31 093717](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/13906df2-7551-45db-bdde-bb8d428a49be)

![Screenshot 2023-08-31 093730](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/4bf7e5ec-187b-4493-b489-8696087e144d)

![Screenshot 2023-08-31 093744](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/f34c1815-c4de-4d5a-be90-7ff48b17c0f8)

![Screenshot 2023-08-31 093759](https://github.com/premalatha-sureshbabu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120620842/b80c1d46-a6ce-4b12-8a95-6c47f41a85ee)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

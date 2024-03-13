# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Udhayanithi M
RegisterNumber: 212222220054 
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data=data.drop(['sl_no','salary'],axis=1)
data.isnull().sum()
data.duplicated().sum()
data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["gender"]=le.fit_transform(data["gender"])
data["ssc_b"]=le.fit_transform(data["ssc_b"])
data["hsc_b"]=le.fit_transform(data["hsc_b"])
data["hsc_s"]=le.fit_transform(data["hsc_s"])
data["degree_t"]=le.fit_transform(data["degree_t"])
data["workex"]=le.fit_transform(data["workex"])
data["specialisation"]=le.fit_transform(data["specialisation"])
data["status"]=le.fit_transform(data["status"])
x=data.iloc[:,:-1]
x
y=data["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy =accuracy_score(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display =metrics.ConfusionMatrixDisplay(confusion_matrix = confusion,display_labels = [True,False])
cm_display.plot()
```

## Output:
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/d84fc56b-c9e6-4989-8b0e-9d4e3e94470e)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/22218e4a-5569-4850-a6da-6e92b50535c2)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/e48ec69d-1cc6-475f-9836-ed2d7369a816)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/2d7d8c93-8a23-40af-8adb-72665661a4b5)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/9e3d8401-3fe4-44fd-81b2-5e1f6f890759)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/cfb71936-5b8f-4fca-b4f9-ec30955cb245)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/141f4843-98d7-4aa0-bf75-2d0a48100028)
![image](https://github.com/UdhayanithiM/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/127933352/93224fb3-4232-44ae-9e6a-f4a48b01c2a0)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

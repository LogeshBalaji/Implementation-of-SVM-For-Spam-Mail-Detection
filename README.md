# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect a labeled dataset of emails, distinguishing between spam and non-spam.
2. Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.
3. Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.
4. Split the dataset into a training set and a test set.
5. Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.
6. Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.
7. Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.
8. Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.
9. Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Logesh B
RegisterNumber:  24900577
*/
```
```.py
import pandas as pd
data = pd.read_csv("spam.csv",encoding = "windows - 1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![SVM For Spam Mail Detection](sam.png)
### data.head()
![Screenshot 2024-12-20 225215](https://github.com/user-attachments/assets/3739e22b-cb94-4885-b8c7-369e8d75bd94)
### data.info()
![Screenshot 2024-12-20 225803](https://github.com/user-attachments/assets/72b4a4d4-9fdd-4693-bf37-412b47d92e12)

### data.isnull().sum()
![Screenshot 2024-12-20 225249](https://github.com/user-attachments/assets/f183daa1-1f56-4886-9941-628fb557bdb6)
### Y_prediction value
![Screenshot 2024-12-20 225456](https://github.com/user-attachments/assets/5488a0d1-763f-4942-8fda-ed4ca25ccb57)
### Accuracy value
![Screenshot 2024-12-20 225527](https://github.com/user-attachments/assets/ebe1d393-5180-454c-ab68-673f8cbbceb6)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

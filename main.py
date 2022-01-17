
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

col_names=['Age','Sex','Chest pain type','BP','Cholesterol','FBS over 120','EKG results','Max HR','Exercise angina','ST depression','Slope of ST','Number of vessels fluro','Thallium','Heart Disease']

# load dataset

values = pd.read_csv("heart_data.csv")
x = values.iloc[1:, :-1]
y = values.iloc[1:, -1]
print(x)
print(y)
x = values.drop('Heart Disease', axis=1)
y = values['Heart Disease']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.25)

ModelName=[]
TestAccuracy=[]
TrainAccuracy=[]
PredictedResult=[]
result=[]
def Train(name,model):
    ModelName.append(name)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    PredictedResult.append(list(y_predicted))
    train_Acc = model.score(x_train, y_train)

    print("Model,",name)
    print('train accuracy for ',name,'=',train_Acc)

    TrainAccuracy.append(train_Acc)
    test_Acc = accuracy_score(y_test, y_predicted)
    TestAccuracy.append(test_Acc)
    print('test accuracy for ',name,'=',test_Acc)
    print('\n')
models=[LogisticRegression(),GaussianNB(),KNeighborsClassifier(n_neighbors=5),svm.SVC(kernel='linear'),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=100)]
names=['logitic','Knife Bias','KNN','SVM','Decision Tree','Random Forest']
import threading
threads = []
t1 = time.perf_counter()

t=0
import time
for i in range(len(names)):
    t = threading.Thread(target=Train,args=(names[i],models[i]))
    t.start()
    time.sleep(0.1)

    threads.append(t)

for thread in threads:
    thread.join()

t2 = time.perf_counter()
print('\n')
print(f'Finished in {t2-t1} seconds')
print(threads)
print(TrainAccuracy)
print(TestAccuracy)
print(len(PredictedResult))
for i in PredictedResult:
    print(i)
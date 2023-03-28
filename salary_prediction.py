import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,silhouette_score
import matplotlib.pyplot as plt



dataset = pd.read_csv("salary.csv")
pd.set_option('display.max_columns',None)
print(dataset.info())
print(dataset.describe())
plt.hist(dataset['marital-status'])
plt.xlabel("marital-status")
plt.ylabel("count")
plt.title("marital-status counts")
plt.show()
dataset['workclass'] = dataset['workclass'].replace([' ?'], ' State-gov')
dataset['native-country'] = dataset['native-country'].replace([' ?'], ' United-States')
label = LabelEncoder()
dataset['workclass'] = label.fit_transform(dataset['workclass'])
dataset['education'] = label.fit_transform(dataset['education'])
dataset['marital-status'] = label.fit_transform(dataset['marital-status'])
dataset['occupation'] = label.fit_transform(dataset['occupation'])
dataset['relationship'] = label.fit_transform(dataset['relationship'])
dataset['race'] = label.fit_transform(dataset['race'])
dataset['sex'] = label.fit_transform(dataset['sex'])
dataset['native-country'] = label.fit_transform(dataset['native-country'])
y = dataset.salary
dataset.drop("salary",axis=1,inplace=True)
dt_model = DecisionTreeClassifier()
svm = SVC()
X1,X2,y1,y2 = train_test_split(dataset,y,test_size=0.3)
dt_model.fit(X1,y1)
dt_res = dt_model.predict(X2)
print("Result for accuracy for decision tree:",accuracy_score(dt_res,y2))
print("Result for f1 score for decision tree:",f1_score(dt_res,y2,average="binary", pos_label=" <=50K"))

svm.fit(X1,y1)
svm_res = svm.predict(X2)
print("Result for accuracy for  SVM:",accuracy_score(svm_res,y2))
print("Result for f1 score for SVM:",f1_score(svm_res,y2,average="binary", pos_label=" <=50K"))
plt.bar(["SVM","Decision Tree"],[accuracy_score(svm_res,y2),accuracy_score(dt_res,y2)])
plt.xlabel("Algorithms")
plt.ylabel("accuracy")
plt.show()

plt.bar(["SVM","Decision Tree"],[f1_score(svm_res,y2,average="binary", pos_label=" <=50K"),f1_score(dt_res,y2,average="binary", pos_label=" <=50K")])
plt.xlabel("Algorithms")
plt.ylabel("F1")
plt.show()
inertia_list = []
for n in range(1,10):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(dataset)
        inertia_list.append(kmeans.inertia_)
        
plt.plot(range(1,10),inertia_list)
plt.xlabel("Cluster")
plt.ylabel("Inertia")
plt.show()
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit(dataset).labels_
print("silhouette_score for KMeans:",silhouette_score(dataset,labels,metric="euclidean",sample_size=1000,random_state=200))







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

# Exploratory Data Analysis

columns = ['id','clumb thickness','UCell Size','UCell Shape','Marginal Adhesion','SCell Size',
           'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

data = pd.read_csv('cancer.data',names=columns)

data.info()

#Attribute Information for Data:
# 1.Sample code number: id number
# 2.Clump Thickness: 1 - 10
# 3.Uniformity of Cell Size: 1 - 10
# 4.Uniformity of Cell Shape: 1 - 10
# 5.Marginal Adhesion: 1 - 10
# 6.Single Epithelial Cell Size: 1 - 10
# 7.Bare Nuclei: 1 - 10
# 8.Bland Chromatin: 1 - 10
# 9.Normal Nucleoli: 1 - 10
# 10.Mitoses: 1 - 10
# 11.Class: (2 for benign, 4 for malignant)

data['Bare Nuclei'] = data['Bare Nuclei'].apply(lambda x: float(x) if x!='?' else None)

data=data.dropna().reset_index(drop=True)
data=data.drop(['id'],axis=1)

# Train Test Split

X = data.drop('Class',axis=1).values
y = data['Class'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42, stratify=y)

# Decide K the number of nearest neighbors
#Setup arrays to store training and test accuracies
neighbors = np.arange(1,10)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 


plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#Check neighbors with GridSearchCV
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
print('The best n neighbor is ' + str(model.best_params_['n_neighbors']) + '.')


#Setup a knn classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

#Calculate confision matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d")
plt.show()


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Check accuracy
accuracy=knn.score(X_test,y_test)
print('k-NN accuracy of our model is equal %' + str(round(accuracy, 2)*100) + '.')

classification_report(y_test, y_pred)

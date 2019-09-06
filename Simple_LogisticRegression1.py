import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report

data=pd.read_csv("/home/neureol/Downloads/diabetes.csv")
print(data)
print(data.columns)
y=data["Outcome"]
X=data[[ "Pregnancies","Glucose", "BloodPressure", "SkinThickness", "Insulin",
       "BMI", "DiabetesPedigreeFunction", "Age"]]
print(X)
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y)
Model = LogisticRegression()
Model.fit(X_train, y_train)
predictions = Model.predict(X_test)
print(predictions)
Accuracy = Model.score( X_test,y_test)
print(Accuracy)
print(classification_report(y_test,predictions))
Confusion_Matrix = metrics.confusion_matrix(y_test, predictions)
print("[[True Postives False Negatives]  [False postives True Negatives]]",Confusion_Matrix)
#Accuracy = (TP+TN)/total


##ROC,AUC

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = metrics.roc_curve(y_test,predictions)


#training set accuracy
training_set_auccuracy = auc(fpr, tpr)

print(training_set_auccuracy)


plt.plot(fpr, tpr, color='darkorange', label='ROC curve ' % training_set_auccuracy)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(Model, open(filename, 'wb'))

# some time later...

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

importing dataset
dataset = pd.read_csv('ans4.csv')
X = dataset.iloc[:,:-5].values
Y = dataset.iloc[:,6].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = .5,random_state = 0)



#classifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

cm
x = cm[0][0] + cm[1][1] + cm[2][2]
total = x + cm[0][1] + cm[0][2] + cm[1][0] + cm[1][2] + cm[2][0] + cm[2][1]
print(x/total)
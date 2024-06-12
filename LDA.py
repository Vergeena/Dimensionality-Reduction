# LDA

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def split_scalar(indep_X, dep_Y):    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size=0.25, random_state=0)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def lcadim(X_train, y_train, n):
    # Applying LDA
    lda = LDA(n_components=n)
    X_train_lda = lda.fit_transform(X_train, y_train)
    explained_variance = lda.explained_variance_ratio_
    return lda, X_train_lda, explained_variance

def cm_prediction(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return classifier, cm, accuracy, report, X_test, y_test

def logistic(X_train, y_train, X_test, y_test):
    # Fitting logistic regression to the Training set
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def svm_linear(X_train, y_train, X_test, y_test):
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def svm_rbf(X_train, y_train, X_test, y_test):
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def naive_bayes(X_train, y_train, X_test, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def knn(X_train, y_train, X_test, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def decision_tree(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    classifier, cm, accuracy, report, X_test, y_test = cm_prediction(classifier, X_test, y_test)
    return classifier, cm, accuracy, report, X_test, y_test

def lda_result(acclog, accsvml, accsvmrbf, accnavie, accknn, accdes, accrf):
    dataframe = pd.DataFrame(index=['LDA_2'], columns=['Logistic', 'SVM_linear', 'SVM_rbf', 'Naive Bayes', 'KNN', 'Decision Tree', 'Random Forest'])
    for number, idx in enumerate(dataframe.index):
        dataframe['Logistic'][idx] = acclog[number]
        dataframe['SVM_linear'][idx] = accsvml[number]
        dataframe['SVM_rbf'][idx] = accsvmrbf[number]
        dataframe['Naive Bayes'][idx] = accnavie[number]
        dataframe['KNN'][idx] = accknn[number]
        dataframe['Decision Tree'][idx] = accdes[number]
        dataframe['Random Forest'][idx] = accrf[number]
        return dataframe

    # Load dataset
dataset = pd.read_csv("Wine.csv")
indep_X = dataset.iloc[:, 0:13].values
dep_Y = dataset.iloc[:, 13].values

# Perform LDA
lda, X_train_lda, explained_variance = lcadim(indep_X, dep_Y, 2)

# Initialize lists to store accuracies
acclog = []
accsvml = []
accsvmrbf = []
accnavie = []
accknn = []
accdes = []
accrf = []

# Split data, train classifiers, and evaluate
X_train, X_test, y_train, y_test = split_scalar(X_train_lda, dep_Y)

classifier, cm, accuracy, report, X_test, y_test = logistic(X_train, y_train, X_test, y_test)
acclog.append(accuracy)

classifier, cm, accuracy, report, X_test, y_test = svm_linear(X_train, y_train, X_test, y_test)
accsvml.append(accuracy)

classifier, cm, accuracy, report, X_test, y_test = svm_rbf(X_train, y_train, X_test, y_test)
accsvmrbf.append(accuracy)

classifier, cm, accuracy, report, X_test, y_test = naive_bayes(X_train, y_train, X_test, y_test)
accnavie.append(accuracy)

classifier, cm, accuracy, report, X_test, y_test = knn(X_train, y_train, X_test, y_test)
accknn.append(accuracy)

classifier, cm, accuracy, report, X_test, y_test = decision_tree(X_train, y_train, X_test, y_test)
accdes.append(accuracy)

classifier, cm, accuracy, report, X_test, y_test = random_forest(X_train, y_train, X_test, y_test)
accrf.append(accuracy)

# Create result dataframe
result = lda_result(acclog, accsvml, accsvmrbf, accnavie, accknn, accdes, accrf)

result
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
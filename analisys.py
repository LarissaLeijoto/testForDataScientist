import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from pandas.tools import plotting
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sys import argv

def main():

    # read the data

    # Take any '.' or '??' values as NA
    data = pd.read_csv("winequality.csv",
            na_values=['.', '??'],
            sep=','
            )
    num_bins = int(argv[1])
    #print "Dataset Lenght:: ", len(data)
    #print "Dataset Shape:: ", data.shape

       
    #Describe data in relation to some metrics
    #print data.describe()
    
    # correlation between the data
    #print data.corr()

    #groupby_Type = data.groupby('type')

    #print groupby_Type.mean()
    #print groupby_Type.min()
    #print groupby_Type.max()

    # Put the target (quality) in another DataFrame
    target = pd.DataFrame(data, columns=["quality"])
    data = data.drop(['quality'],  axis=1)

    #Discretization
    type_ = {'White': 1,'Red': 2} 
    data.type = [type_[item] for item in data.type]

    #Normalization
    #min_max_scaler = MinMaxScaler()
    #MinMaxScaler(copy=True, feature_range=(0, 1))
    #x_scaled = min_max_scaler.fit_transform(data)
    #df = pd.DataFrame(x_scaled)

    # transform the dataset with KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=num_bins, encode='ordinal').fit(data)
    X_binned = enc.transform(data)
    #np.savetxt("foo.csv", X_binned, header="type,fixed acidity,volatile acidity,citric acid, residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol", delimiter=",")
      

    target_ =  target.values.ravel()

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_binned, target_, test_size=0.3) # 70% training and 30% test

    #Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
   
    
   

    #X_train, X_test, y_train, y_test = train_test_split(X_binned, target_, test_size=0.20)  

    #classifier = DecisionTreeClassifier()  
    #classifier.fit(X_train, y_train) 

    #y_pred = classifier.predict(X_test)  

    #print(confusion_matrix(y_test, y_pred))  
    #print(classification_report(y_test, y_pred)) 

    #Feature subset selection 
    #forest = ExtraTreesClassifier(n_estimators=50)
    #forest = forest.fit(df, target_)
    #print forest.feature_importances_ 
    #model = SelectFromModel(forest, prefit=True)
    #X_new = model.transform(df)
    
    #print X_new
    n_classes = target.shape
    print n_classes
    
    X_train, X_test, y_train, y_test = train_test_split(X_binned, target_, test_size=0.2)
    
    #svr_rbf = SVC(kernel='rbf', C=1e3, gamma=0.1)
    #y_rbf = svr_rbf.fit(X_new, target_).predict(X_new)
    #svr_rbf.fit(X_train, y_train)
    #confidence = svr_rbf.score(X_test, y_test)
    #print(confidence)  
    n_classes = 10
    C_2d_range = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    gamma_2d_range = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            for k in ['linear','poly','rbf','sigmoid']:
                for k in ['poly', 'rbf']:
                        clf = svm.SVC(kernel=k, C=C, gamma=gamma)
                        clf.fit(X_train, y_train)
                        y_score = clf.decision_function(X_test)
                        # For each class
                        precision = dict()
                        recall = dict()
                        average_precision = dict()
                        for i in range(n_classes):
                                precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
                                average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

                        # A "micro-average": quantifying score on all classes jointly
                        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                        y_score.ravel())
                        average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                                        average="micro")
                        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                        .format(average_precision["micro"]))
                    #confidence = clf.score(X_test, y_test)
                    #print(C, gamma, k, y_score)
    '''
    '''
    
    #importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    #indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    #print("Feature ranking:")

    #for f in range(df.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    #plt.figure()
    #plt.title("Feature importances")
    #plt.bar(range(df.shape[1]), importances[indices],
    #    color="r", yerr=std[indices], align="center")
    #plt.xticks(range(df.shape[1]), indices)
    #plt.xlim([-1, df.shape[1]])
    #plt.show()
    
    
    #print target
    #print data
    #print len(target)
    # describe data in relation to some metrics
    #print data.describe()
    # correlation between the data
    #print data.corr()

    #groupby_Type = data.groupby('type')

    #print groupby_Type.mean()
    #print groupby_Type.min()
    #print groupby_Type.max()

   

    #print y_rbf
    #groupby_Type.boxplot()
    '''

if __name__ == '__main__':
    main()


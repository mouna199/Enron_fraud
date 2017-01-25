#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi",'salary', 'deferral_payments', 'total_payments', 'loan_advances','bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',"shared_receipt_with_poi"]

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
data_dict.pop("TOTAL",0)

"""
#detecting outliers
import matplotlib.pyplot
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
new_one=sorted(data_dict, key=lambda x: (data_dict[x]['salary']),reverse=True)

for i in data_dict:
    salary = data_dict[i]["salary"]
    bonus = data_dict[i]["bonus"]
    if salary != 'NaN' and bonus != 'NaN' and salary >= 100000 or bonus >=500000:
        print(i)


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

"""

### Task 3: Create new feature(s)
for keys, values in data_dict.iteritems():
    if values["to_messages"] != "NaN" and \
                    values["from_messages"] != "NaN" and \
                    values["from_poi_to_this_person"] != "NaN" and \
                    values["from_this_person_to_poi"] != "NaN":

        values["to_person"] = 100*float(values["from_poi_to_this_person"]) / \
                                       values["to_messages"]
        values["from_person"] = 100*float(values["from_this_person_to_poi"]) / \
                                             values["from_messages"]
    else:
        values["to_person"] = 0
        values["from_person"] = 0
features_list.append("to_person")
features_list.append("from_person")
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels f rom dataset for local testing
#we split the data into features and labels
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# use Kbest to select the best features 
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(k=14)
selected_features = kbest.fit_transform(features,labels)
features_selected=[features_list[i+1] for i in kbest.get_support(indices=True)]
scores = kbest.scores_
features_rank = sorted(zip(features_list[1:], scores), key = lambda l: l[1],\
     reverse = True)
print features_rank
print 'Features selected by SelectKBest:'
print features_selected
# we only take out one feature to avoid correlatio and work with the rest of selected features
features_list=['poi','salary', 'total_payments', 'loan_advances', 'bonus',
               'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
               'other', 'long_term_incentive', 'restricted_stock', 'from_poi_to_this_person',
               'from_person']

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Naive bayes 
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
#we split the data 
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.30, random_state=42)

target_names = ["Not POI", "POI"]

#working with the first classifier, naive bayes
NB_clf = GaussianNB()
NB_clf.fit(features_train,labels_train)
accuracy=NB_clf.score(features_test,labels_test)
print "accuracy",accuracy
pred=NB_clf.predict(features_test)
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)
######################################################################################################

#### the second classifier is Decision Tree
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

scaled = ()
pca = PCA()
scaled= MinMaxScaler()
combined_transformers = FeatureUnion([ ("scale",scaled),("pca", pca)])
tree=tree.DecisionTreeClassifier()
estimators = [("features", combined_transformers), ('tree',tree )]
pipe= Pipeline(estimators)

parameters = {'tree__max_features':['auto', 'sqrt', 'log2'], 'tree__splitter':['best'],'tree__class_weight':['balanced'],
               'tree__criterion':['gini'],'tree__min_samples_leaf':[2], 'tree__min_samples_split': [3,4,5,6,7,8],
              "features__pca__n_components": [2]}
cv = StratifiedShuffleSplit(labels,100, random_state = 42)
clf = GridSearchCV(pipe, parameters,n_jobs=-1,verbose=10,scoring='precision',cv=cv)

clf =clf.fit(features, labels)
clf = clf.best_estimator_
print clf
from tester import test_classifier
print ' '
# use test_classifier to evaluate the model selected by GridSearchCV
print "Tester Classification report"
test_classifier(clf, data_dict, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)

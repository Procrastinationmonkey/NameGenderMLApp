import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer


# Load data
df = pd.read_csv('C:\Users\Dell\Desktop\\names.csv')

df2=df
df2.sex.replace({'F':0,'M':1},inplace=True)

X_1 =df2['name']


# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(X_1)

cv.get_feature_names()

from sklearn.model_selection import train_test_split

# Features 
X
# Labels
y = df2.sex

#Splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")



# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

# Vectorize features function
features = np.vectorize(features)


from sklearn.feature_extraction import DictVectorizer

df_X = features(df2['name'])

df_y = df2['sex']

dv = DictVectorizer()


dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.2, random_state=30)

dfX_train

dv = DictVectorizer()

dv.fit_transform(dfX_train)

#Applying decision tree classifier

from sklearn.tree import DecisionTreeClassifier
 
dtree = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dtree.fit(my_xfeatures, dfy_train)


testname = ["Jawad"]
transform_dv =dv.transform(features(testname))

vec = transform_dv.toarray()


dtree.predict(vec)

if dtree.predict(vec) == 0:
    print("Female")
else:
    print("Male")
    


print(dtree.score(dv.transform(dfX_train), dfy_train))


# Accuracy on test set
print(dtree.score(dv.transform(dfX_test), dfy_test))



from sklearn.externals import joblib

decisiontreeModel = open("C:\Users\Dell\Desktop\decisiontreemodel.pckl","wb")

joblib.dump(dtree,decisiontreeModel)

decisiontreeModel.close



NaiveBayesModel = open("C:\Users\Dell\Desktop\\naivebayesgendermodel.pkl","wb")

joblib.dump(clf,NaiveBayesModel,)

NaiveBayesModel.close()
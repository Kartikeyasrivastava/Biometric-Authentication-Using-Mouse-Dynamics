from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
import extractor
from sklearn import svm




'''

Generation of training data from the text files.
This data contains the 62 features with around 1200 Key strokes i.e; rows.
They are stored in the format of

'''

X_nokku = extractor.features(13,"nokku/")
X_nokku_train = X_nokku[:int(X_nokku.shape[0]*0.8)]
X_nokku_test = X_nokku[int(X_nokku.shape[0]*0.8):]

X_renu = extractor.features(13,"renu/")
X_renu_train = X_renu[:int(X_renu.shape[0]*0.8)]
X_renu_test = X_renu[int(X_renu.shape[0]*0.8):]


X_bapi = extractor.features(13,"bapi/")
X_bapi_train = X_bapi[:int(X_bapi.shape[0]*0.8)]
X_bapi_test = X_bapi[int(X_bapi.shape[0]*0.8):]


X_uday = extractor.features(13,"uday/")
X_uday_train = X_uday[:int(X_uday.shape[0]*0.8)]
X_uday_test = X_uday[int(X_uday.shape[0]*0.8):]

X_ramola = extractor.features(13,"ramola/")
X_ramola_train = X_ramola[:int(X_ramola.shape[0]*0.8)]
X_ramola_test = X_ramola[int(X_ramola.shape[0]*0.8):]


X_santhosh = extractor.features(13,"santosh/")
X_santhosh_train = X_santhosh[:int(X_santhosh.shape[0]*0.8)]
X_santhosh_test = X_santhosh[int(X_santhosh.shape[0]*0.8):]

print("Completed data gathering")


'''
Application of one-class-svm algorithm, using rbf kernel.
Here the data is trained on user-data of nokku.
The y_name_train indicates the output of data

'''

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_nokku)



y_nokku_train= clf.predict(X_nokku_train)

print("nokku_test:")
print (y_nokku_train)

y_santhosh = clf.predict(X_santhosh)
print("\ny_santhosh_test:")
print (y_santhosh)

y_ramola = clf.predict(X_ramola)
print("\nramola_test")
print (y_ramola)


y_renu = clf.predict(X_renu)
print("\nrenu:")
print (y_renu)

y_bapi = clf.predict(X_bapi)
print("\nbapi:")
print (y_bapi)

y_uday = clf.predict(X_uday)
print("\nuday:")
print (y_uday)
print("\nfinal result:")

#Training classifier
X_train = np.concatenate((X_nokku_train,X_bapi,X_renu,X_santhosh,X_ramola,X_uday),axis=0)
y_train = np.concatenate((y_nokku_train,y_bapi,y_renu,y_santhosh,y_ramola,y_uday),axis=0)



#Using KNn classifier for finding the accuracy of the file
#By adjusting the k_features we can specify the required no.of features

knn = KNeighborsClassifier(n_neighbors=63)
sfs1 = SFS(knn,
           k_features=3,
           forward=True,
           floating=False,
           scoring='accuracy',
           cv=4)

#Here sfs1.k_feature_idx_  gives the index of the final selected features

sfs1 = sfs1.fit(X_train, y_train)
X_train_final = X_train[:,sfs1.k_feature_idx_]
X_nokku_test_final = X_nokku_test[:,sfs1.k_feature_idx_]
clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf2.fit(X_train_final,y_train)

y_nokku_test = clf2.predict(X_nokku_test_final)
print("\nAccuracy:")

#print(float(y_nokku_test[y_nokku_test == -1].size))
#print (float(y_nokku_test.size))

print(1-(float(y_nokku_test[y_nokku_test == -1].size)/float(y_nokku_test.size)))



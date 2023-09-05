from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from classes.sample import Sample
import joblib




predictSample = []

B = Sample("./out/blaise_processed.wav")
JM = Sample("./out/julien_processed.wav")
YM = Sample("./out/marguet_processed.wav")
V = Sample("./out/valerie_processed.wav")
Y = Sample("./out/yacine_processed.wav")


merged, target = Sample.concat()

x_train, x_test, y_train, y_test = train_test_split(merged, target)
clf = svm.SVC(probability=True)
clf.fit(x_train, y_train)




res = clf.predict(x_test)
res = res.tolist()


print("Accuracy was : ", metrics.accuracy_score(y_test, res))

joblib.dump(clf, "./model/model.pkl") 
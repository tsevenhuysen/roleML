from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pandas as pd

print("Loading dataset.")
df = pd.read_csv("dataset_participants2.csv")

target = df["position"]
data = df.drop(["position","participantId","gameId"], axis=1)

print("Splitting dataset.")
shuffle_split = StratifiedShuffleSplit(train_size=0.9, n_splits=10)

print("Creating Random Forest classifier.")
c = RandomForestClassifier(min_samples_split=5, n_estimators=100)

accs = []

print("Fitting trees.")
for train_index, test_index in shuffle_split.split(data, target):
    c.fit(data.iloc[train_index], target.iloc[train_index])
    accs.append(accuracy_score(target[test_index], c.predict(data.iloc[test_index])))

print("Dumping .sav to exploration and roleml folders.")
joblib.dump(c, "hive_role_identification_model.sav")
joblib.dump(c, "../roleml/role_identification_model.sav")

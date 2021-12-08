import pandas as pd 

penguins = pd.read_csv('penguins_cleaned.csv')

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

df['species'] = df.species.map(target_mapper)

X = df.drop('species', axis=1)
y = df['species']


# funing parammeters isn't our main focus, so we will keep it simple as it is
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,y)

import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))

print("file saved")
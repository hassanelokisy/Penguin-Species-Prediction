import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write("""

# Penguin Species Prediction App
This is a very simple project, just testing Heroku.

""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

uploaded_file = st.sidebar.file_uploader(
    'Upload your input csv file', type=['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features_from_sidebar():
        island = st.sidebar.selectbox('Islan', options=[
            'Biscoe', 'Dream', 'Torgersen'
        ])
        sex = st.sidebar.selectbox('Sex', ['male', 'female'])
        bill_length_mm = st.sidebar.slider(
            'Bill length (mm)', 32.1, 59.6, 50.0)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            'Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider(
            'Body mass (g)', 2700.0, 6300.0, 4207.0)
        print(type(sex), type(island), type(bill_length_mm), type(flipper_length_mm), type(body_mass_g))
        data = {
            'sex': sex,
            'island': island,
            'bill_length_mm': float(bill_length_mm),
            'bill_depth_mm': float(bill_depth_mm),
            'flipper_length_mm': float(flipper_length_mm),
            'body_mass_g': float(body_mass_g),
        }

        return pd.DataFrame(data, index=[0])

    input_df = user_input_features_from_sidebar()
    print(input_df)



penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'], axis=1)
df = pd.concat([input_df,penguins],axis=0)


encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1]

st.subheader("User Input features")

st.dataframe(df)



clf = pickle.load(open('penguins_clf.pkl', 'rb'))

prediction = clf.predict(df)
predicted_proba = clf.predict_proba(df)


st.header('Prediction:')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.success(penguins_species[prediction][0])

st.text('')
st.text('')
st.text('')

st.subheader('Prediction Probability for all calsses')
d = pd.DataFrame(predicted_proba, columns=['Adelie','Chinstrap','Gentoo'], index=['Probability'] )
st.write(d)

"""Class demo: AM Tue 27-Feb-2024"""

import pickle

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


def load_classifiers():
    clf_nn = tf.keras.models.load_model('heart_failure.h5')
    with open('svm_linear_heart.pkl', 'rb') as f:
        clf_svl = pickle.load(f)
    return clf_nn, clf_svl


def heart_prediction(input_):
    input_arr = np.asarray(input_)
    input_arr = input_arr.reshape(1, -1)

    # load classifiers
    clf_nn, clf_svl = load_classifiers()

    # make predictions
    pred_nn = clf_nn.predict(input_arr)
    pred_svl = clf_svl.predict(input_arr)

    # return predictions
    return pred_nn, pred_svl


def setup_page():
    # st.title('Heart Failure Prediction')
    st.set_page_config(
        page_title='Heart Failure Prediction',
        layout='wide'
    )
    image = Image.open('heart.png')
    st.image(image, use_column_width=False)

    st.write('Enter the following details to predict whether a patient has heart failure or not.')

    ## variable inputs
    age = st.number_input('Age of the patient:',min_value=0, step=1)
    anaemia = st.number_input('Anaemia | yes or no | yes = 1 and no = 0:',min_value=0, step=1)
    creatinine_phosphokinase = st.number_input('Level of the CPK enzyme in the blood (mcg/L):',min_value=0, step=1)
    diabetes = st.number_input('Diabetes | yes or no | yes = 1 and no = 0:',min_value=0, step=1)
    ejection_fraction = st.number_input('Percentage of blood leaving the heart at each contraction:',min_value=0, step=1)
    high_blood_pressure = st.number_input('Hypertension | yes or no | yes = 1 and no = 0:',min_value=0, step=1)
    platelets = st.number_input('Platelet count of blood (kiloplatelets/mL):',min_value=0, step=1)
    serum_creatinine = st.number_input('Level of serum creatinine in the blood (mg/dL):',min_value=0.00, step=0.01)
    serum_sodium = st.number_input('Level of serum sodium in the blood (mEq/L):',min_value=0, step=1)
    sex = st.number_input('Sex | male or female | male = 1 and female = 0:',min_value=0, step=1)
    smoking = st.number_input('Habit of smoking | yes or no | yes = 1 and no = 0:',min_value=0, step=1)
    time = st.number_input('Follow-up period (days):',min_value=0, step=1)

    inputs = [
        age,
        anaemia,
        creatinine_phosphokinase,
        diabetes,
        ejection_fraction,
        high_blood_pressure,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
        time,
    ]

    predict = ''
    if st.button('Predict'):
        pred_nn, pred_svl = heart_prediction(inputs)
        st.write(f'NN Prediction: {pred_nn}')
        st.write(f'SVM Prediction: {pred_svl}')


def main():
    setup_page()


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE 
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def predict_tab(model, scaler, label_encoder, df_subset_ct):
    # Input data
    age = st.number_input("Age", min_value=0.0)

    sex_options = df_subset_ct['Sex'].unique().tolist() 
    sex = st.selectbox("Select Sex", sex_options)

    race_options = df_subset_ct['Race'].unique().tolist() 
    race = st.selectbox("Select Race", race_options)

    death_city_options = sorted(df_subset_ct['DeathCity'].unique().tolist())  
    death_city = st.selectbox("Select Death City", death_city_options)
    
    # Memproses data yang di-input
    if st.button("Predict"):
        # Create DataFrame from input
        df_new = pd.DataFrame([[age, sex, race, death_city]], columns=['Age', 'Sex', 'Race', 'DeathCity'])


        cat_col = ['Sex','Race','DeathCity']
        for col in cat_col:
            label_encoder.fit(df_subset_ct[col])
            
            df_new[col] = label_encoder.transform(df_new[col])

        # Scale the numerical and categorical columns
        df_new_scaled = scaler.transform(df_new[['Age', 'Sex', 'Race', 'DeathCity']])

        # Make prediction
        predicted_cocaine = model.predict(df_new_scaled)
        
        # Convert prediction to boolean
        predicted_bool = True if predicted_cocaine[0] == 1 else False
        
        # Display result
        st.subheader(f"Predicted: {predicted_cocaine[0]:.2f}")
        st.write(f"Is cocaine involved? {'Yes' if predicted_bool else 'No'}")

def train_and_model(df_subset_ct):
    # Encode
    df_encode = df_subset_ct.copy()
    label_encoder = preprocessing.LabelEncoder()
    cat_col = ['Sex','Race','DeathCity']


    df_encode.loc[:,cat_col] = df_encode.loc[:,cat_col].apply(label_encoder.fit_transform)
    

    # Scaling
    scaler = RobustScaler()
    df_r_scale = df_encode.copy()
    df_r_scale[['Age','Sex','Race','DeathCity']] = scaler.fit_transform(df_r_scale[['Age','Sex','Race','DeathCity']])

    ### Evaluate
    x = df_r_scale.iloc[:,0:-1].values
    y = df_r_scale.iloc[:,-1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    smote = SMOTE(random_state=42)

    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    model = LogisticRegression()
    model.fit(x_train_resampled, y_train_resampled)

    # Prediksi
    y_pred = model.predict(x_test)

    return model, scaler, label_encoder



def preprocessed(df_csv):
    df_subset= df_csv[['Age', 'Sex','Race','DeathCity','ResidenceState','Cocaine']]
    df_subset_ct_predrop = df_subset[df_subset['ResidenceState'] == 'CT']
    df_subset_ct = df_subset_ct_predrop.drop(columns = 'ResidenceState')
    df_subset_ct = df_subset_ct.astype({'Age': int}) 
    cat_col = ['Sex','Race']
    simpleImputer = SimpleImputer(strategy='most_frequent')

    for i in cat_col:
        # Store the column
        df_cat_copy = df_subset_ct[i].copy() 

        # Fit transform
        simple_imputed_data = simpleImputer.fit_transform(df_cat_copy.values.reshape(-1, 1))

        # Replace the data in the column
        df_subset_ct.loc[:, i] = simple_imputed_data.flatten()


    
    df_subset_ct.loc[:, 'Age'] = winsorize(df_subset_ct['Age'], limits=[0, 0.01])

    return df_subset_ct
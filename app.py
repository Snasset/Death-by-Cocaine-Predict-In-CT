import streamlit as st
import numpy as np
import pandas as pd
import prediction_tab
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


st.title('Death by Cocaine in Connecticut')
st.write('''The rate of overdose deaths in Connecticut increased. In this case, Cocaine. 
    In Connecticut, if you are found guilty of cocaine possession, for a first offense, 
    you face up to seven years in prison as well as a maximum \\$50,000 fine.
    For a second offense, now you might spend up to 15 years in prison and a maximum \\$100,000 fine.
    In short, Cocaine is illegal in Connecticut.
    So, this model will predict if a person is involved in cocaine-related death based on race, age, sex, and death city.
        ''')



def main():
    try:
        df_csv = pd.read_csv("drug_deaths.csv")

        st.write(df_csv)

        df_subset_ct = prediction_tab.preprocessed(df_csv)
        model, scaler, label_encoder = prediction_tab.train_and_model(df_subset_ct)
        # Membuat tab dashboard
        tab1, tab2 = st.tabs(["Predict involved in cocaine-related", "Analyze and Model"])
        
        # Tab 1
        with tab1:
            prediction_tab.predict_tab(model, scaler, label_encoder, df_subset_ct)
        
        # Tab 2
        with tab2:
           dataUnderstanding(df_csv)
           dataPrep(df_csv)

    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan path sudah benar.")




def dataUnderstanding(df_csv):
    # Take all the needed columns
    df_subset= df_csv[['Age', 'Sex','Race','DeathCity','ResidenceState','Cocaine']]
    st.write(''' ## Data Understanding

            The dataset provide a variant of drugs death. In this case, only need one drug which is Cocaine.

            The following attributes will be used in the analysis:

            - Age
            - Sex
            - Race
            - DeathCity
            - ResidenceState
            - Cocaine
             ''')
    st.write("### Subset Data:")
    st.write('In this section, i will analyze which attributes/columns will be removed.')
    st.write('These are the nessecary columns')
    st.write(df_subset.head())

    # Mengambil data berdasarkan nilai CT (Connecticut)
    st.write('Statistic Description')
    st.write(df_subset.describe(include='all'))
    st.divider()


def dataPrep(df_csv):
    # Sum of missing datas
    st.write("## Data Preperation:")
     # Take all the needed columns
    df_subset= df_csv[['Age', 'Sex','Race','DeathCity','ResidenceState','Cocaine']]

    # Mengambil data berdasarkan nilai CT (Connecticut)
    df_subset_ct_predrop = df_subset[df_subset['ResidenceState'] == 'CT']
    df_subset_ct = df_subset_ct_predrop.drop(columns = 'ResidenceState')
    df_subset_ct = df_subset_ct.astype({'Age': int}) 
    st.write("### Subset Data CT:")
    st.write('Take all the residence state in CT and then remove ResidenceState. Because all the data will always in CT.')
    st.write('Data')
    st.write(df_subset_ct.head())
    st.write('Statistic Description')
    st.write(df_subset_ct.describe(include='all'))
    # Imputing
    # Perform imputation to a categorical column with SimpleImputer
    cat_col = ['Sex','Race']
    simpleImputer = SimpleImputer(strategy='most_frequent')

    # Loop every categorical calumn and impute
    for i in cat_col:
        # Store the column
        df_cat_copy = df_subset_ct[i].copy() 

        # Fit transform
        simple_imputed_data = simpleImputer.fit_transform(df_cat_copy.values.reshape(-1, 1))

        # Replace the data in the column
        df_subset_ct.loc[:, i] = simple_imputed_data.flatten()
    
    # Check if there any missing value
    st.write("### Using SimpleImputer to impute missing values:")
    st.write(df_subset_ct.isna().sum())

    st.write("### Outliers:")
    
    df_outlier = df_subset_ct.select_dtypes(exclude=['object']) 
    for column in df_outlier:
        plt.clf()
        plt.figure(figsize=(10, 2))
        sns.boxplot(data=df_outlier, x=column)
        plt.tight_layout()
        st.pyplot(plt)
    df_subset_ct.loc[:, 'Age'] = winsorize(df_subset_ct['Age'], limits=[0, 0.01])
    st.write("Since there's only one outlier, I'm using the Winsorizing method to handle it by applying limits of 0 and 0.01.")

    st.write("### Result:")
    plt.figure(figsize=(10, 2))
    sns.boxplot(data=df_subset_ct, x='Age')
    plt.tight_layout()
    st.pyplot(plt)  

    st.write("### Visualization:")

    # DEATHCITY
    st.write("#### By DeathCity:")
    death_city_count = df_subset_ct.groupby('DeathCity').sum()['Cocaine'].sort_values(ascending=False)

    plt.figure(figsize=(10,5))

    myColors = sns.color_palette('pastel')[0:5]
    death_city_count[0:20].plot(kind='bar',color= myColors)
    plt.ylabel('Jumlah Kasus Cocaine')
    plt.xlabel('Kota Tempat Kematian')
    plt.title('Jumlah Kasus Cocaine Terkait Kematian di Setiap Kota di Connecticut')
    st.pyplot(plt)  


    # AGE
    st.write("#### By Age:")
    age_count = df_subset_ct.groupby('Age').sum()['Cocaine'].sort_values(ascending=False)

    plt.figure(figsize=(10,5))

    myColors = sns.color_palette('pastel')[0:5]
    age_count[0:20].plot(kind='bar',color= myColors)
    plt.ylabel('Jumlah Kasus Cocaine')
    plt.xlabel('Umur')
    plt.title('Jumlah Kasus Cocaine Terkait Umur di Connecticut')
    st.pyplot(plt)  


    # RACE
    st.write("#### By Race:")
    race_count = df_subset_ct.groupby('Race').sum()['Cocaine'].sort_values(ascending=False)

    plt.figure(figsize=(10,5))

    myColors = sns.color_palette('pastel')[0:5]
    race_count[0:20].plot(kind='bar',color= myColors)
    plt.ylabel('Jumlah Kasus Cocaine')
    plt.xlabel('Ras')
    plt.title('Jumlah Kasus Cocaine Terkait Ras di Connecticut')
    st.pyplot(plt)  


    # SEX
    st.write("#### By Sex:")
    sex_count = df_subset_ct.groupby('Sex').sum()['Cocaine'].sort_values(ascending=False)

    plt.figure(figsize=(10,5))

    myColors = sns.color_palette('pastel')[0:5]
    sex_count[0:20].plot(kind='bar',color= myColors)
    plt.ylabel('Jumlah Kasus Cocaine')
    plt.xlabel('Jenis Kelamin')
    plt.title('Jumlah Kasus Cocaine Terkait Jenis Kelamin di Connecticut')
    st.pyplot(plt)  
    plt.clf()

    # Piechart
    st.write("#### Cocaine involved in Connecticut:")
    cocaine_counts = df_subset_ct['Cocaine'].value_counts()

    cocaine_counts = cocaine_counts.reindex([0, 1], fill_value=0)

    pieChartLabels = ['No','Yes']

    myColors = sns.color_palette('pastel')
    plt.pie(cocaine_counts, labels = pieChartLabels, colors = myColors, autopct='%.0f%%')
    st.pyplot(plt)  


    st.divider()

    modelling(df_subset_ct)

def modelling(df_subset_ct):
    # Encode
    st.write("## Modelling")
    st.write("#### Encode with LabelEncoder")
    df_encode = df_subset_ct.copy()
    cat_col = ['Sex','Race','DeathCity']

    label_encoder = preprocessing.LabelEncoder()

    df_encode.loc[:,cat_col] = df_encode.loc[:,cat_col].apply(label_encoder.fit_transform)
    st.write(df_encode.head())

    # Scaling
    st.write("#### Scaling with RobustScaler")
    scaler = RobustScaler()
    df_r_scale = df_encode.copy()
    df_r_scale[['Age','Sex','Race','DeathCity']] = scaler.fit_transform(df_r_scale[['Age','Sex','Race','DeathCity']])
    st.write(df_r_scale.head())

    ### Model
    st.write("### Logistic Regression")
    x = df_r_scale.iloc[:,0:-1].values
    y = df_r_scale.iloc[:,-1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



    logis_reg = LogisticRegression(solver='lbfgs', max_iter=1000)

    logis_reg.fit(x_train, y_train)
    y_pred = logis_reg.predict(x_test)

    st.write("#### Prediction Data(5):")
    logr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
    st.write(logr_diff.head())

    target_names = ['Not Cocaine', 'By Cocaine']
    # Menampilkan classification report sebagai DataFrame
    classification_report_df = metrics.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_df).transpose()

    # Tampilkan DataFrame di Streamlit
    st.write("### Classification Report:")
    st.dataframe(classification_report_df)

    # Menampilkan akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### Accuracy:", round(accuracy, 3))


    # Confussion matrix untuk melihat masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih
    st.write("#### Confussion Matrix")
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix)
                , annot=True
                , cmap="YlGnBu"
                ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)
    st.write("It shows that the class for cocaine is imbalance for 0 an 1 as seen on the piechart before")

    
    st.divider()
    ### Evaluate
    st.write("### Evaluate")
    st.write("#### Oversampling with SMOTE")
    st.write("So, to balance between class. I'll use SMOTE")
    x = df_r_scale.iloc[:,0:-1].values
    y = df_r_scale.iloc[:,-1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    smote = SMOTE(random_state=42)

    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    ## Inisialisasi dan latih model
    model = LogisticRegression()
    model.fit(x_train_resampled, y_train_resampled)

    # Prediksi
    y_pred = model.predict(x_test)

    st.write("#### Prediction Data(5):")
    logr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
    st.write(logr_diff.head())


    target_names = ['Not Cocaine', 'By Cocaine']
    # Menampilkan classification report sebagai DataFrame
    classification_report_df = metrics.classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_df).transpose()

    # Tampilkan DataFrame di Streamlit
    st.write("### Classification Report:")
    st.dataframe(classification_report_df)

    # Menampilkan akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### Accuracy:", round(accuracy, 3))

    # Confussion matrix untuk melihat masalah klasifikasi machine learning dimana keluaran dapat berupa dua kelas atau lebih
    st.write("#### Confussion Matrix")
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    class_names=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix)
                , annot=True
                , cmap="YlGnBu"
                ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

    

if __name__ == "__main__":
    main()
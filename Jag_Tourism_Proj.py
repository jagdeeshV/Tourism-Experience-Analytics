# Tourism Proejct in full through Menu
#  Created by Jagadeesh V during third week of Feb. 2025
#    Packaages for all
import streamlit as st
import pandas as pd
import numpy as np
import keyboard
import time
import os
import psutil

st.title('Tourism Experience Analytics')
st.write("1. Datasets cleansing and Merging, 2. ML Metrics & Charts, 3. Classificatiom Model - Visit Mode Prediction, 4. Regression Model - Rating Prediction & 5. recommendation - Attractions")

st.markdown(
    """
    <style>
    .rlg1 {
        margin-bottom: 0.1rem; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'one_time' not in st.session_state:
    st.session_state.one_time = 0
if st.session_state.one_time == 0:
    st.session_state.one_time = 1
    #######  ----------  Clensing and Merging Data  ----------  #######
    # 1A. Cleansing and Merging user related data 
    st.subheader('1. Datasets cleansing and Merging')
    st.markdown('<p class="rlg1">1A. Merging User Demographic datasets</p>', unsafe_allow_html=True)
    st.markdown('<p class="rlg1">Datasets - 1. User, 2. Continent, 3. Region, 4. Country, 5. City & 6. Location \n\n</p>', unsafe_allow_html=True)

    df1 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Continent.xlsx')
    df2 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Region.xlsx')
    df_Merge1 = pd.merge(df2, df1, how='left', left_on='ContentId', right_on = 'ContenentId')
    st.markdown('<p class="rlg1">Merged Continent and Region</p>', unsafe_allow_html=True)

#    st.markdown('<p class="rlg1"></p>', unsafe_allow_html=True)

    df1 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Country.xlsx')
    df_Merge2 = pd.merge(df1, df_Merge1, on='RegionId', how='left')
    st.markdown('<p class="rlg1">Merged Country</p>', unsafe_allow_html=True)
    df1 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\City.xlsx')
    df_Merge1 = pd.merge(df1, df_Merge2, on='CountryId', how='left')
    st.markdown('<p class="rlg1">Merged City</p>', unsafe_allow_html=True)

    df_Merge1['FullId'] = df_Merge1['ContenentId'].astype(str) + df_Merge1['RegionId'].astype(str) + df_Merge1['CountryId'].astype(str) + df_Merge1['CityId'].astype(str)
    df_Merge1['FullId'] = df_Merge1['FullId'].astype("float")
    df_Merge1.to_csv('D:\Guvi\Tourism Proj\Datasets\Location.csv', index=False)
    st.markdown('<p class="rlg1">Merged Location</p>', unsafe_allow_html=True)

    df1 = pd.read_excel("D:\Guvi\Tourism Proj\Datasets\Tuser.xlsx")
    df1.ffill(inplace=True)
    df1['FullId'] = df1['ContenentId'].astype(str) + df1['RegionId'].astype(str) + df1['CountryId'].astype(str) + df1['CityId'].astype(str)
    df1['FullId'] = df1['FullId'].astype("float")
    df_MergeUL = pd.merge(df1, df_Merge1, on='FullId', how='inner')
    df_MergeUL.to_csv('D:\Guvi\Tourism Proj\Datasets\TUserLoc.csv', index=False)
    st.markdown('<p class="rlg1">Merged User with Demographics and created TUserLoc.csv</p>', unsafe_allow_html=True)

    # 1B. Cleansing and Merging Transaction related data 
    st.write('')
    st.markdown('<p class="rlg1">B. Merging Attraction and Visit detail datasets"</p>', unsafe_allow_html=True)
    st.write('<p class="rlg1">Datasets - 1. Transaction, 2. Item, 3. Type & 4. Mode \n</p>', unsafe_allow_html=True)

    df1 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Item.xlsx')
    df2 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Type.xlsx')
    df_Merge1 = pd.merge(df1, df2,on='AttractionTypeId', how='left')
    st.markdown('<p class="rlg1">Merged Item & Type</p>', unsafe_allow_html=True)

    df1 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Transaction.xlsx')
    df_Merge2 = pd.merge(df1, df_Merge1, on='AttractionId', how='left')
    st.markdown('<p class="rlg1">Merged Transaction</p>', unsafe_allow_html=True)

    df1 = pd.read_excel('D:\Guvi\Tourism Proj\Datasets\Mode.xlsx')
    df_Merge1 = pd.merge(df_Merge2, df1, how='left', left_on='VisitMode', right_on = 'VisitModeId')
    st.markdown('<p class="rlg1">Merged Visit Details\n</p>', unsafe_allow_html=True)

    df_Merge1.to_csv('D:\Guvi\Tourism Proj\Datasets\Tran.csv', index=False)
    st.write('Merged and created Tran.csv')
    # Merging both sets to single consolidated.csv
    df_Merge2 = pd.merge(df_Merge1, df_MergeUL, on='UserId', how='inner')
    df_Merge2.to_csv('D:\Guvi\Tourism Proj\Datasets\Consolidated.csv', index=False)
    st.write("C. Merged both A & B as consolidated.csv")

### ------------------------------------------------------------------------------------------------------------------------------------ ###

# Define the functions for each Menu optins (Classification, Regression & Recommendation
def Proc_Classfy():
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBClassifier, plot_importance
    from sklearn.metrics import accuracy_score, classification_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import keyboard
    import os
    import psutil
    import sys

    # 1. Functions & class called by its Main Streamlit routine
    def load_and_preprocess_data(csv_file):
        # Load data
        merged_data = pd.read_csv(csv_file)
        # Feature engineering
        merged_data['VisitSeason'] = pd.to_datetime(merged_data['VisitYear'].astype(str) + '-' + merged_data['VisitMonth'].astype(str) + '-01').dt.month.map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
        days = '01'
        merged_data['VisitSeason1'] = pd.to_datetime(dict(year=merged_data.VisitYear, month=merged_data.VisitMonth, day=days)).dt.strftime('%Y-%b-%d')
        return merged_data

    def month_to_numeric(month_abbr):
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        try:
            return months.index(month_abbr.capitalize()) + 1
        except ValueError:
            return 0

    class TourClassifyClass:
    # 1A. Class variable initialization 
        def __init__(self):
            self.le = LabelEncoder()
            # Create preprocessing pipeline
            numeric_features = ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 'Rating']
            categorical_features = ['VisitSeason', 'ContenentId_x', 'RegionId_x', 'CountryId_x', 'CityId_x', 'AttractionTypeId']

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
            ])
            
            # Create preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
        
        # Create a pipeline with preprocessor and model
            self.model = Pipeline(steps=[('preprocessor', self.preprocessor),
                                ('classifier', XGBClassifier(random_state=42))])

        def prepare_features(self, data):
            try:
                # Prepare features and target
                features = ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 'VisitSeason', 'ContenentId_x', 'RegionId_x', 'CountryId_x', 'CityId_x', 'AttractionTypeId', 'Rating']
                
                X = data[features]
                y = data['VisitMode_y']

                # Encode target variable
                y = self.le.fit_transform(y)
                return X,  y
            except Exception as e:
                st.error(f"Error in feature preparation: {str(e)}")
                print(f"Error in feature preparation: {str(e)}")
                return None, None

        def train_classification_model(self, x, y):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Perform cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=10)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)

            # Calculate feature importance
            feature_importance = self.model.named_steps['classifier'].feature_importances_
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            
            return X_test, y_test, y_pred, cv_scores, feature_importance, feature_names

        def plot_feature_importance(self, feature_importance, feature_names):
            # Sort features by importance
            indices = np.argsort(feature_importance)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.barh(range(len(feature_importance)), feature_importance[indices])
            plt.yticks(range(len(feature_importance)), [feature_names[i] for i in indices])
            plt.xlabel("Importance")
            plt.tight_layout()
            return plt

    ##### --------------------------------------------------------------------------------------------------------------------------------------- #####
    # 2. Main Streamlit routine Accepting user inputs, calling functions & Predicting
        def main_ui(self, csv_file):
            try:
                st.subheader('3. Visit Mode Prediction [Classification  Model]')
                st.write(f"\nCleaned Data set Loaded : {csv_file}")
                temp_msg = st.empty()
                temp_msg.text("\n Wait. Preparing features and training the dataset")
                if 'one_time' not in st.session_state:
                    st.session_state.one_time = 0
                if csv_file is not None:
                    data = load_and_preprocess_data(csv_file)
                    if data is not None and not data.empty:
                        x, y = self.prepare_features(data)
                        if x is not None and y is not None:
                            if st.session_state.one_time == 0:
                                st.session_state.one_time = 1
                                X_test, y_test, y_pred, cv_scores, feature_importance, feature_names = self.train_classification_model(x, y)
        
                                # Print cross-validation results
                                st.write(f"Cross-validation scores: {cv_scores}")
                                st.write(f"Mean CV score: {cv_scores.mean():.4f}")
                                st.write('')
                                st.write(f"Predict {y_pred}")
                                
                                # Print classification report
                                st.write("\nClassification Report:")
                                st.write(classification_report(y_test, y_pred, target_names=self.le.classes_))
        
                                # Plot feature importance
                                plt = self.plot_feature_importance(feature_importance, feature_names)
                                fig = plt.show()
                                st.pyplot(fig)
                                st.write('\n\nEnd')
                            df = data
                            uniq_users = df[['Contenent', 'Region', 'Country', 'CityName', 'UserId']].drop_duplicates()
                            uniq_users['combined_det'] = uniq_users.apply(lambda row: f"{row['Contenent']} ~~ {row['Region']} ~~ {row['Country']} ~~ {row['CityName']} ~~ {row['UserId']}", axis=1)
                            selected_user_det = st.selectbox('Select user Demographic details:', uniq_users['combined_det'])
                            sel_user = selected_user_det.split(' ~~ ')
                            selected_user_row = df[(df['Contenent'] == sel_user[0]) & 
                                          (df['Region'] == sel_user[1]) & 
                                          (df['Country'] == sel_user[2]) & 
                                          (df['CityName'] == sel_user[3]) & 
                                          (df['UserId'] == int(sel_user[4]))]

                            filtered_df = df[df['UserId'] == int(sel_user[4])]
                            filtered_df = filtered_df[['AttractionType', 'Attraction', 'AttractionAddress', 'AttractionId']]
                            uniq_attr = filtered_df.drop_duplicates()
                            uniq_attr['combined_det'] = uniq_attr.apply(lambda row: f"{row['AttractionType']} ~~ {row['Attraction']} ~~ {row['AttractionAddress']} ~~ {row['AttractionId']}", axis=1)
                            selected_attr_det = st.selectbox('Select a Attraction Details:', uniq_attr['combined_det'])
                            sel_attr = selected_attr_det.split(' ~~ ')
                            selected_attr_row = df[(df['AttractionType'] == sel_attr[0]) & 
                                          (df['Attraction'] == sel_attr[1]) & 
                                          (df['AttractionAddress'] == sel_attr[2]) & 
                                          (df['AttractionId'] == int(sel_attr[3]))]

                            uniq_dt = df[['VisitSeason1']].drop_duplicates()
                            selected_season = st.selectbox('Select Visit Season:', uniq_dt)
                            sel_season = selected_season.split('-')

                            if st.button('Predict Visit Mode'):
                                if st.session_state.one_time == 1:
    # 2E1 Subsequent times Model training o clicking predct button to avoid processing time on each and every input
                                    X_test, y_test, y_pred, cv_scores, feature_importance, feature_names = self.train_classification_model(x, y)
                                input_data = pd.DataFrame({
                                    "UserId":  [int(sel_user[4])],
                                    "VisitYear": [sel_season[0]],
                                    "VisitMonth": [month_to_numeric(sel_season[1])],
                                    "AttractionId": [int(sel_attr[3])],
                                    "ContenentId_x":  [df.loc[df['Contenent'] ==  sel_user[0], 'ContenentId_x'].values[0]],
                                    "RegionId_x": [df.loc[df['Region'] ==  sel_user[1], 'RegionId_x'].values[0]],
                                    "CountryId_x": [df.loc[df['Country'] ==  sel_user[2], 'CountryId_x'].values[0]],
                                    "CityId_x": [df.loc[df['CityName'] ==  sel_user[3], 'CityId_x'].values[0]],
                                    "Attraction": [sel_attr[1]],
                                    "AttractionTypeId": [df.loc[df['AttractionType'] ==  sel_attr[0], 'AttractionTypeId'].values[0]],
                                    "VisitSeason": [df.loc[df['VisitSeason1'] ==  selected_season, 'VisitSeason'].values[0]],
                                    "Rating": [df.loc[(df['UserId'] ==  int(sel_user[4])) & (df["AttractionId"] == int(sel_attr[3])), 'Rating'].values[0]],
                                    'VisitMode_y':  [df.loc[(df['UserId'] == int(sel_user[4])) & (df["AttractionId"] == int(sel_attr[3])), 'VisitMode_y'].values[0]]
                                })
                                y_pred = self.model.predict(input_data)
                                Visit_out = self.le.inverse_transform([y_pred])
                                st.write(f"Predicted Visit Mode Name: {y_pred} / {Visit_out}")
                                st.write('')
            except Exception as e:
                st.error(f"Error in Streamlit app: {str(e)}")
                print(f"Error in Streamlit app:  {str(e)}")

    # Invoking the execution
    Task_run = TourClassifyClass()
    Task_run.main_ui(csv_file)
### ------------------------------------------------------------------------------------------------------------------------------------ ###

def Proc_Regrress():
    # Packages for Regression
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
    import pickle

    # Loading, Preprocessing, Training, etc functions called by Main Streamlit routine
    class TourRegrressClass:
        # 1A. Class variable initialization 
        def __init__(self):
            self.salient_features = [
                "VisitYear", "VisitMonth", "VisitMode_y",
                "AttractionType", "Attraction", "Contenent", "Region",  "Country", "CityName"
            ]
            # Create model pipeline
            self.rf_model = RandomForestRegressor(
                    n_estimators=50,  # Increased number of trees
                    max_depth=5,      # Limited depth to prevent overfitting
                    random_state=42,
                    n_jobs=-1  # Use all CPU cores
                )
            
        def load_and_preprocess_data(self, csv_file):
            # Load data
            merged_data = pd.read_csv(csv_file)
            # Feature engineering
            merged_data['VisitSeason'] = pd.to_datetime(merged_data['VisitYear'].astype(str) + '-' + merged_data['VisitMonth'].astype(str) + '-01').dt.month.map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
            days = '01'
            merged_data['VisitSeason1'] = pd.to_datetime(dict(year=merged_data.VisitYear, month=merged_data.VisitMonth, day=days)).dt.strftime('%Y-%b-%d')
            return merged_data

        def prepare_features(self, data):
            # Create preprocessing pipeline

            X = data[self.salient_features]
            y = data["Rating"]

            #Categorical to Numeric data conversion
            X = pd.get_dummies(X, drop_first=True)

            return X, y

        def train_model(self, X, y):
            #Train / Test data splitting
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            #XGBOOST Model 
            XGB_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            XGB_model.fit(X_train, y_train)

            #Decision Tree model
            dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
            dt_model.fit(X_train, y_train)

            #DC Tree metrics evaluation 
            dt_pred = dt_model.predict(X_test)
            dt_pred = np.round(dt_pred).astype(int)
            dt_pred = np.clip(dt_pred, 1, 5) 

            dt_mae = mean_absolute_error(y_test, dt_pred)
            dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

            dt_r2 = r2_score(y_test, dt_pred)  

            #Random Forest Model Training and metrics evaluation
            self.rf_model.fit(X_train, y_train)

            rf_pred = self.rf_model.predict(X_test)
            rf_pred = np.round(rf_pred).astype(int)
            rf_pred = np.clip(rf_pred, 1, 5)  

            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

            rf_r2 = r2_score(y_test, rf_pred)

            # Saving the RF trained model as pickle file for further predictions
            joblib.dump(self.rf_model,r"D:\Guvi\Tourism Proj\Datasets\\rating_rf_model.pkl")

            return dt_pred, dt_mae, dt_rmse, dt_r2, rf_pred, rf_mae, rf_rmse, rf_r2

        def month_to_numeric(self, month_abbr):
            # Alpha Month to numeric conversion 
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            try:
                return months.index(month_abbr.capitalize()) + 1
            except ValueError:
                return 0
    ##### --------------------------------------------------------------------------------------------------------------------------------------- #####

    # 2. Main Streamlit routine Accepting user inputs, calling functions & Predicting
        def main_ui(self, csv_file):
            try:
                st.subheader("4. Rating - Prediction [Regression Model]")
                df = self.load_and_preprocess_data(csv_file)
                if 'one_time' not in st.session_state:
                    st.session_state.one_time = 0
                if st.session_state.one_time == 0:
                    st.session_state.one_time = 1
                    if df is not None and not df.empty:
                        x, y = self.prepare_features(df)
                        if x is not None and y is not None:
                            dt_pred, dt_mae, dt_rmse, dt_r2, rf_pred, rf_mae, rf_rmse, rf_r2 = self.train_model(x, y)

                            text_msg = "Decision Tree Regressor Results \n \
        1. MAE: "+ str(dt_mae)+ ",     2. RMSE: "+ str(dt_rmse)+ ",\n   3. R² Score: "+ str(dt_r2)+ ",  \n \n \
        Random Forest Regressor Results \n \
        1. MAE: "+ str(rf_mae)+ ",      2. RMSE: "+ str(rf_rmse)+ ",\n    3. R² Score: "+ str(rf_r2)
                            st.text_area("aa", text_msg, disabled=True, height = 200, label_visibility="collapsed")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.histplot(df["Rating"], bins=5, kde=True, color="blue")
                            ax.set_title("Distribution of Ratings")
                            st.pyplot(fig)

                with open("D:\Guvi\Tourism Proj\Datasets\\rating_rf_model.pkl", 'rb') as file:
                    data_pkl = pickle.load(file)

                uniq_locn = df[['Contenent', 'Region', 'Country', 'CityName']].drop_duplicates()
                uniq_locn['combined_det'] = uniq_locn.apply(lambda row: f"{row['Contenent']} ~~ {row['Region']} ~~ {row['Country']} ~~ {row['CityName']}", axis=1)
                selected_locn_det = st.selectbox('Select user Demographic details:', uniq_locn['combined_det'])
                sel_locn = selected_locn_det.split(' ~~ ')
                selected_user_row = df[(df['Contenent'] == sel_locn[0]) & 
                              (df['Region'] == sel_locn[1]) & 
                              (df['Country'] == sel_locn[2]) & 
                              (df['CityName'] == sel_locn[3])]

                uniq_attr = df[["AttractionType",'Attraction']].drop_duplicates()
                uniq_attr['combined_det'] = uniq_attr.apply(lambda row: f"{row['AttractionType']} ~~ {row['Attraction']}", axis=1)
                selected_attr_det = st.selectbox('Select a Attraction Details:', uniq_attr['combined_det'])
                sel_attr = selected_attr_det.split(' ~~ ')
                selected_attr_row = df[(df['AttractionType'] == sel_attr[0]) & 
                              (df['Attraction'] == sel_attr[1])]

                uniq_dt = df[['VisitSeason1', 'VisitMode_y']].drop_duplicates()
                uniq_dt['combined_det'] = uniq_dt.apply(lambda row: f"{row['VisitSeason1']}-{row['VisitMode_y']}", axis=1)
                selected_season = st.selectbox('Select Visit Season & Mode:', uniq_dt['combined_det'] )
                sel_season = selected_season.split('-')
                selected_season_row = df[(df['VisitSeason1'] == sel_season[0]+'-'+sel_season[1]) & 
                              (df['VisitMode_y'] == sel_season[3])]
                if st.button('Predict Rating'):
                    if data_pkl.ndim != 1:
                        raise ValueError("The loaded ndarray is not a 1D array.")
                    element_names = data_pkl.tolist()
                    full_values = {name: False for name in element_names}
                    
                    mmyy_vals = {
                        'VisitYear': sel_season[0],
                        'VisitMonth': self.month_to_numeric(sel_season[1])
                    }
                    cn = "CityName_"+ sel_locn[3]
                    cntry = "Country_"+ sel_locn[2]
                    rgn = "Region_"+ sel_locn[1]
                    contn = "Contenent_"+ sel_locn[0]
                    attr = "Attraction_"+ sel_attr[1]
                    attr_typ = "AttractionType_"+ sel_attr[0]
                    visitmod = 'VisitMode_y_'+ sel_season[3]
                    input_vals = {
                        cn: True,
                        cntry: True,
                        rgn: True,
                        contn: True,
                        attr: True,
                        attr_typ: True,
                        visitmod: True
                    }
                    full_values.update(mmyy_vals)
                    full_values.update(input_vals)
                    input_data = pd.DataFrame([full_values])

                    rf_model = joblib.load(r"D:\Guvi\Tourism Proj\Datasets\\rating_rf_model.pkl")
                    
                    rf_pred = rf_model.predict(input_data)
                    rf_pred = np.round(rf_pred).astype(int)
                    rf_pred = np.clip(rf_pred, 1, 5)  
                    st.markdown(f"### Predicted Rating : {rf_pred}")
                    st.write('')
            except Exception as e:
                st.error(f"Error in Streamlit app: {str(e)}")
                print(f"Error in Streamlit app:  {str(e)}")
    ##### --------------------------------------------------------------------------------------------------------------------------------------- #####

    # Invoking the execution
    Task_run = TourRegrressClass()
    Task_run.main_ui(csv_file)

def Proc_Recommend():
    # 4 of 4 Recommendation part.  3 Types  Collaborative Filtering,   Content based Filtering  &  Hybrid  recommendations
    # Exclusive packages for Recommendation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    st.subheader("5. Attractions Suggestion  [Recommendation Model]")

    df = pd.read_csv(csv_file)

    # Loading Dataset & selecting required columns
    selected_cols = ['UserId', 'AttractionId', 'Attraction', 'AttractionType', 'VisitMode_y', 'Rating']
    df = df[selected_cols]

    # Handling Missing Values
    df.dropna(inplace=True)

    # Prepare Data for Surprise Library
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['UserId', 'AttractionId', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train SVD Model
    svd = SVD()
    svd.fit(trainset)

    # Functions for each model
    def collaborative_recommend(user_id, df, model, top_n=5):
    # Collaborative Filtering Recommendation
        user_attractions = df[df['UserId'] == user_id]['AttractionId'].unique()
        all_attractions = df['AttractionId'].unique()
        unseen_attractions = [a for a in all_attractions if a not in user_attractions]
        
        predictions = [(a, model.predict(user_id, a).est) for a in unseen_attractions]
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        df1 = df[df['AttractionId'].isin([rec[0] for rec in recommendations])][['Attraction', 'AttractionType']]
        unique_df =df.groupby('Attraction').first().reset_index()
        df_result = unique_df[['Attraction', 'AttractionType', 'VisitMode_y', 'Rating']].head(10)
        return df_result

    def content_based_recommend(user_attractions, df, top_n=5):
    # Content based Filtering
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['AttractionType'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        attraction_indices = df[df['Attraction'].isin(user_attractions)].index.tolist()
        scores = similarity_matrix[attraction_indices].mean(axis=0)
        
        recommended_indices = scores.argsort()[-top_n:][::-1]
        return df.iloc[recommended_indices][['Attraction', 'AttractionType']]

    def hybrid_recommend(user_id, df, model, top_n=15):
    ### HYBRID RECOMMENDATION ###
        content_rec = content_based_recommend(df[df['UserId'] == user_id]['Attraction'].tolist(), df, top_n=top_n)
        collab_rec = collaborative_recommend(user_id, df, model, top_n=top_n)
        
        hrec = pd.concat([content_rec, collab_rec]).drop_duplicates().head(top_n)
        hybrid_rec = hrec.dropna().dropna(axis=1).head(5)
        return hybrid_rec

    # Main UI -> Getting user input and recommending
    uniq_users = df[['UserId']].drop_duplicates()
    uniq_users = uniq_users.sort_values(by=['UserId'])
    
    uniq_attr = df[['Attraction']].drop_duplicates()
    uniq_attr = uniq_attr.sort_values(by=['Attraction'])

    uniq_attr_typ = df[['AttractionType']].drop_duplicates()
    uniq_attr_typ = uniq_attr_typ.sort_values(by=['AttractionType'])

    V_userid = st.selectbox('Select User Id', uniq_users['UserId'])
    V_attraction  = st.selectbox('Select a Attraction', uniq_attr['Attraction'])
    V_attractiontype = st.selectbox('Select a Attraction type', uniq_attr_typ['AttractionType'])

    st.write("\nCollaborative Filtering 10 Recommendations:")
    st.write(collaborative_recommend(V_userid, df, svd))

    st.write("Content-Based top 5 Recommendations:")
    st.write(content_based_recommend([V_attraction, V_attractiontype], df))

    st.write("\nHybrid top 5 Recommendations:")
    st.write(hybrid_recommend(V_userid, df, svd))

# Create a main menu
st.sidebar.title("Main Menu")
menu = st.sidebar.radio("Select an Option", ("Visit Mode Prediction [Classificatiom Model]", "Rating Prediction [Regression Model]", "Suggesting Attractions [Recommendation]", "Exit Application"), index=None )

# Call the selected program
csv_file = "D:\Guvi\Tourism Proj\Datasets\Consolidated.csv"
if menu == "Visit Mode Prediction [Classificatiom Model]":
    Proc_Classfy()
elif menu == "Rating Prediction [Regression Model]":
    Proc_Regrress()
elif menu == "Suggesting Attractions [Recommendation]":
    Proc_Recommend()
elif menu == "Exit Application":
    st.subheader("### Thank You for using the App")
    time.sleep(3)
    # Close streamlit browser tab
    keyboard.press_and_release('ctrl+w')
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
    st.stop()
    close_script = """
        <script type="text/javascript">
            window.close();
        </script>
    """
    st.markdown(close_script, unsafe_allow_html=True)


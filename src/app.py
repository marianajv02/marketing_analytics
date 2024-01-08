# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 05:46:30 2024

@author: Mariana & Sihan
"""

import json
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui
import joblib
import seaborn as sns
import numpy as np
from shiny import App, ui, render, reactive
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.linear_model import RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_curve, precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.patches as mpatches
import pickle


#Train set
df_train = pd.read_csv('clean_train_data.csv')
#Test set
df_New = pd.read_csv('new_data.csv')

#Import training models
with open('classification_to_cluster.pkl', 'rb') as file:
    mk = pickle.load(file) 
with open('kmeans9.pkl', 'rb') as file:
    Kmeans = pickle.load(file)
    
reg = []
for i in range(9):
    model_name = f'reg_{i}.pkl'
    with open(model_name, 'rb') as file:
        loaded_model = pickle.load(file)
    reg.append(loaded_model)



keep_cols = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE','CREDIT_LIMIT']

scaler = StandardScaler()
X_data = scaler.fit_transform(df_train[keep_cols])
df_train['kmeans_9'] = KMeans(n_clusters =9, n_init = 'auto', random_state= 123).fit_predict(X_data)

#Classification of new data into clusters
#transformers
numeric_transf = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='median')), ('scaler',StandardScaler())]
)
# Setup the preprocessing steps
preprocess = ColumnTransformer(
    transformers = [
        ('num', numeric_transf, selector(dtype_exclude='object'))
    ]
)
def create_features(df):
    df['purch_per_trx'] = np.where(df['PURCHASES_TRX']>0,df['PURCHASES']/df['PURCHASES_TRX'], 0)
    df['cash_adv_per_trx'] = np.where(df['CASH_ADVANCE_TRX']>0,df['CASH_ADVANCE']/df['CASH_ADVANCE_TRX'], 0)
    df['BALANCE_by_freq'] = df['BALANCE']*df['BALANCE_FREQUENCY']
    df['ONEOFF_PURCHASES_by_freq'] = df['ONEOFF_PURCHASES']*df['ONEOFF_PURCHASES_FREQUENCY']
    df['INSTALLMENTS_PURCHASES_by_freq'] = df['INSTALLMENTS_PURCHASES']*df['PURCHASES_INSTALLMENTS_FREQUENCY']
    df['CASH_ADVANCE_by_freq'] = df['CASH_ADVANCE']*df['CASH_ADVANCE_FREQUENCY']
    df['interest_p_year'] = np.where(df['TENURE']>0,(df['BALANCE'] - ((df['PURCHASES'] + df['CASH_ADVANCE']) - (df['PAYMENTS'] + df['MINIMUM_PAYMENTS']) )) /df['TENURE'],(df['BALANCE'] - (df['PURCHASES'] + df['CASH_ADVANCE']) - (df['PAYMENTS'] + df['MINIMUM_PAYMENTS']) ))
    return df
keep_cols1 = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
keep_cols_engi2 = ['PURCHASES', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE', 'purch_per_trx',
       'cash_adv_per_trx', 'BALANCE_by_freq', 'ONEOFF_PURCHASES_by_freq',
       'INSTALLMENTS_PURCHASES_by_freq', 'CASH_ADVANCE_by_freq','interest_p_year']

df = create_features(df_train)
X = df[keep_cols1]

#kmeans prediction

y = df['kmeans_9']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 123)

y_pred = mk.predict(X_test)
best_n = 9

X = df_New[keep_cols1]
y_pred_new = mk.predict(X) #m is the cluster model
df_New['cluster'] = y_pred_new

#Credit limit prediction
params = {'n_estimators': 600,
 'min_samples_split': 16,
 'max_depth': 12,
 'learning_rate': 0.01,
 "loss": "huber",}

target = 'CREDIT_LIMIT'
X = df[keep_cols1]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 123)

def predict_credit(data, cluster):
    if isinstance(data, pd.Series):
        data = data.to_frame().T
    model = reg[cluster]
    credit_limit = model.predict(data)
    return credit_limit

df_predict = df_New[keep_cols1]
for i in range(len(df_New)):
    predicted_credit = predict_credit(df_predict.loc[i], df_New.loc[i, 'cluster'])
    df_New.at[i, 'CREDIT_PREDICTED'] = predicted_credit
    
    
#Setting palette for plots
custom_palette = sns.color_palette('tab20', n_colors=9)  
sns.set_palette(custom_palette) 

    
#Dashboard interface
app_ui = ui.page_fluid(
    
    ui.navset_tab(
        ui.nav("Results and predictor",
                  
               ui.row(
                   ui.h2("Customer classifier and credit limit simulator")  
                ),
                
               ui.row(
                    ui.h4(
                    '''The model considers a dataset containing information about 8,800 customers with different spending habits. The aim is to classify these customers into groups and predict a credit limit for potential new customers'''),
                    ),
               ui.row(
                    ui.h4(
                    '''Customers where classified into the folowing 9 segments. To see more details on the model construction, please refer to page 2'''),
                    ),
               ui.row(
                    ui.column(12,
                              ui.output_plot("cluster_descriptions")
                              )
            
                ),
                   
               ui.h2("Current customer's data"), 
               ui.row(
                    ui.column(4,
                              ui.h4("Number of customers per segment"),
                              ui.output_plot("histogram")
                              ),
                    ui.column(4,
                              ui.h4("Percentage of customers per segment"),
                              ui.output_plot("pie_chart")
                              ),
                    ui.column(4,
                              ui.h4("Credit limit distribution accross segments"),
                              ui.output_plot("boxplot")
                              ),
                    ),
               ui.row(
                    ui.h2("Potential new customer's data"),
            
                    ui.column(4,
                              ui.h4("Number of customers per segment "),
                              ui.output_plot("test_histo")
                              ),
                    ui.column(4,
                              ui.h4("Percentage of customers per segment "),
                              ui.output_plot("test_pie")
                              ),
            
                    ui.column(4,
                              ui.h4("Credit limit distribution accross segments"),
                              ui.output_plot("boxplot_test")
                              ) 
            
            
                    ),
               ui.h2("Deep dive into number of customers by segment and feature (New customers)"), 
                
               ui.row(
                    ui.h4(
                    ''' Please use the bars below to select the feature and segment you wanna review'''),
                    ),
                
             
               #agregar informacion por cluster con resumen para poder cambiar el cluster y el feature
               
               
               ui.row(
                    ui.column(4,
                              ui.input_select("segment", "Segment:", choices=list(range(9))),
                              ui.input_select("feature", "Feature:", choices=keep_cols),
                                 ),
                    ui.column(4,
                              ui.output_plot("detail_plot")
                              ) 
                        ),
                  
               
                 
                
                
                # Single prediction input by user : Poner en otra pestaña
               ui.h2("Single Prediction Input"),
               ui.row(
                    ui.h4(
                    ''' Please change the parameters below to get a prediction on a new customer's segment and credit limit. Beware to select parameters within each feature's range. When you're ready, click on the calculate button at your right'''),
                    ),
                
               ui.row(
                    ui.column(4,
                              
                              ui.input_numeric("BALANCE", "Account balance",1666),
                              ui.input_numeric("BALANCE_FREQUENCY", "Balance Frequency",0.63),
                              ui.input_numeric("PURCHASES", "Amount spent on purchases",1499),
                              ui.input_numeric("ONEOFF_PURCHASES", "Maximum amount spent on a single purchase",1499),
                              ui.input_numeric("INSTALLMENTS_PURCHASES", "Amount spent on installment purchases", 0),
                              ui.input_numeric("CASH_ADVANCE", "Cash Advance", 205),
                              ui.input_numeric("PURCHASES_FREQUENCY", "Frequency of user's purchases", 0.08),
                              ui.input_numeric("ONEOFF_PURCHASES_FREQUENCY", "Frequency of user's on off purchases", 0.08),
                              ui.input_numeric("PURCHASES_INSTALLMENTS_FREQUENCY", "Frequency of user's installments purchases", 0)
                              ),
                    ui.column(4,
                              
                              ui.input_numeric("CASH_ADVANCE_FREQUENCY", " Frequency of cash advances", 0.08),
                              ui.input_numeric("CASH_ADVANCE_TRX", "Number of cash advance transactions by the user", 1),
                              ui.input_numeric("PURCHASES_TRX", "Number of purchases transactions by the user", 1),
                              ui.input_numeric("PAYMENTS", "Amount paid by the user towards the credit card balance", 0),
                              ui.input_numeric("MINIMUM_PAYMENTS", "Minimum amount paid by the user", 253),
                              ui.input_numeric("PRC_FULL_PAYMENT", "Percentage of the full payment made by the user", 0),
                              ui.input_numeric("TENURE", "Tenure of the user's credit card service usage", 12),
                              
                              
                        ),
                    ui.column(4,
                              ui.input_action_button('btn',"Calculate", class_='btn-primary'),
                              ui.output_text_verbatim('txt', placeholder=True)
                              
                        ),
                    ),
               ),
               #TAB 2 details
               ui.nav("Details",
                      ui.row(
                          ui.h2("Relationship between Credit Limit and Other Features"),
                      ),
                      ui.row(
                          ui.column(
                              4,
                              ui.input_select("feature2", "Feature:", choices=keep_cols),
                              
                              ),
                          
                          ui.column(
                              8,
                              ui.output_plot("plot1", click=True, dblclick=True, hover=True, brush=True),ui.column(12, ui.output_text_verbatim("click_info")),
                              ui.column(12, ui.output_text_verbatim("hover_info")),
                              
                      ),
                          ),
                   ),
               ),
  
                )

   


def server(input, output, session):
    


    @output
    @render.plot
    def histogram():
        sns.countplot(x='kmeans_9', data=df_train)
        plt.title('Distribution of customers accross segments')
        plt.xlabel('Cluster')
        plt.ylabel('Counts')
        return plt.gcf()
    
    @output
    @render.plot
    def pie_chart():
        counts = df_train['kmeans_9'].value_counts()
        counts = counts.sort_index()
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(edgecolor='black'))
        plt.title('Distribution of customers accross segments')
        return plt.gcf()
    
    @output
    @render.plot
    def boxplot():
        sns.boxplot(x='kmeans_9', y ='CREDIT_LIMIT', data=df_train)
        return plt.gcf()
    

   
    @output
    @render.plot  
    def test_histo():
        sns.countplot(x='cluster', data=df_New)
        plt.title('Distribution of customers accross segments')
        plt.xlabel('Cluster')
        plt.ylabel('Counts')
        return plt.gcf()
    
    @output
    @render.plot
    def test_pie():
        counts = df_New['cluster'].value_counts()
        counts = counts.sort_index()
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(edgecolor='black'))
        plt.title('Distribution of customers accross segments')
        return plt.gcf()
    

    @output
    @render.plot
    def boxplot_test():
        sns.boxplot(x='cluster', y ='CREDIT_PREDICTED', data=df_New)
        # Set the y-axis limits
        plt.ylim(0, 30000)
        
        # Add labels and title
        plt.xlabel('Cluster')
        plt.ylabel('CREDIT_PREDICTED')
        plt.title('Boxplot of CREDIT_PREDICTED Across Clusters')
        
        return plt.gcf()
    
    @output
    @render.plot
    def detail_plot():
        selected_segment = float(input.segment())
        selected_feature = input.feature()
    
        filtered_data = df_New[df_New['cluster'] == selected_segment]
        
        
        
        sns.histplot(data=filtered_data, x=selected_feature)
        plt.title(f"Histogram of {selected_feature} for Segment {selected_segment}")
        plt.xlabel(selected_feature)
        plt.ylabel('Counts')
        return plt.gcf()


    
    @output
    @render.plot    
    def cluster_descriptions():
        cluster_descriptions = {
                        
            1: 'Cluster 0:  Elders: Lowest balance and balance frequency, low purchases , low cash advance, high tenure',
            2: 'Cluster 1: Cash advance kings: Highest and most frequent cash in advance and high account balance with payment compliance. Low purchases.',
            3: 'Cluster 2:  Cheap installment purchasers: Low balance and frequent purchases in installments',
            4: 'Cluster 3: Cash advance peasant: Cash advance preference (but not as high as segment 6), minimum payments',
            5: 'Cluster 4: Installment purchasers: High balance, high purchases, preference for installments, highest tenure',
            6: 'Cluster 5: One off purchasers: Frequent purchases and one off preference, ',
            7: 'Cluster 6:  Amateurs: Recent customers (low tenure) with low balance and low payment amount',
            8: 'Cluster 7: Low usage: Low purchase amount, few installments and frequent changing balance',
            9: 'Cluster 8: One off high amount: Highest frequency purchases, one off payments preference with high',
            }
            
            # Create a legend-like plot with cluster colors and descriptions
        patches = [mpatches.Patch(color=custom_palette[i - 1], label=cluster_descriptions[i]) for i in cluster_descriptions]
        
        plt.figure(figsize=(10, 5))
        plt.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.5))
        
        # Hide the axes
        plt.axis('off')
        
        return plt.gcf()
        
    

    @reactive.Effect
    @reactive.event(input.btn)
    def predict():
        # Access input values using input object and cast to float
        data = [
            float(input.BALANCE()), float(input.BALANCE_FREQUENCY()),
            float(input.PURCHASES()), float(input.ONEOFF_PURCHASES()),
            float(input.INSTALLMENTS_PURCHASES()), float(input.CASH_ADVANCE()),
            float(input.PURCHASES_FREQUENCY()), float(input.ONEOFF_PURCHASES_FREQUENCY()),
            float(input.PURCHASES_INSTALLMENTS_FREQUENCY()), float(input.CASH_ADVANCE_FREQUENCY()),
            float(input.CASH_ADVANCE_TRX()), float(input.PURCHASES_TRX()),
            float(input.PAYMENTS()), float(input.MINIMUM_PAYMENTS()),
            float(input.PRC_FULL_PAYMENT()), float(input.TENURE())
        ]
    
        columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
                   'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                   'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                   'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
                   'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS', 'MINIMUM_PAYMENTS',
                   'PRC_FULL_PAYMENT', 'TENURE']
    
        df_input = pd.DataFrame([data], columns=columns)
        cluster_num = mk.predict(df_input)  # columns ok
        prediction = predict_credit(df_input, cluster_num[0])  # check columns and input
    
        @output
        @render.text
        def txt():
            return f"Customer belongs to group {cluster_num[0]} with an estimated credit limit of {prediction[0]}"

# Page 2: Model construction details
    @output
    @render.text
    def feature_importance():
        return f'From the plot, we can conclude that for each feature, which clusters are affected more based on the length of the bar\nwe can also get the conclusion the effect is negative or positive'
    @output
    @render.text
    def cluster_decision():
        return f"From two plots for silohuette score and inertia, we can both conclude that cluster 9 is a good choice"
    @output
    @render.text
    def cluster():
        return f"For the test dataset, we don't have dataset belonging to cluster 4. So we don't provide the plot here"
    @output
    @render.plot(alt="A scatterplot") #Plotting all scatterplots for visualizations
    def plot1():
        selected_feature=input.feature2()
        fig, ax = plt.subplots()
        plt.title(f"{selected_feature} vs CREDIT LIMIT")
        ax.scatter(df_train[selected_feature], df_train["CREDIT_LIMIT"])
        return fig
       
        
    

    @output
    @render.text()
    def click_info():
        try:
            selected_feature= input.feature()
            click_json_string = json.dumps(input.plot1_click(), indent=2)
            click_data = json.loads(click_json_string)
            x_value = click_data['x']
            y_value = click_data['y']
            return f'{selected_feature}:{x_value}, \nCREDIT LIMIT: {y_value}'
            

        except Exception as e:
            return "Please click one point, you can see the specific values"
   
    @output
    @render.text()

    def hover_info(): # I killed this part hehe sorry, feel free to use your old code, but i think we can create a small df with two columns : feature and description and filter the description to show in the return using selected_feature
        selected_feature= input.feature()
        x_var = selected_feature  # Replace with the actual x variable in your data
        y_var = "CREDIT_LIMIT"  # Replace with the actual y variable in your data
        correlation = df_train[x_var].corr(df_train[y_var])
        correlation_info = f"Correlation: {correlation:.4f}"
        return f"{correlation_info}\nBALANCE: Remaining balance in the card"    



app = App(app_ui, server, debug=True)






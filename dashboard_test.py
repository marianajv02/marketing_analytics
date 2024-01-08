# Note: This app uses a development version of plotnine.

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui
import joblib
import seaborn as sns
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import numpy as np
from shiny import App, ui, render
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

keep_cols1 = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']


#silouete score and inertia processing
scaler = StandardScaler()
X_data = scaler.fit_transform(df_train[keep_cols])
min_clusters = 3
max_clusters = 30
innertia_metric =[]
silohuette_metric =[]
for k in range(min_clusters, max_clusters + 1):

  labels = Kmeans.fit_predict(X_data)
  silohuette_metric.append(silhouette_score(X_data, labels))
  innertia_metric.append(Kmeans.inertia_)





app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style(
            """
        /* Smaller font for preformatted text */
        pre, table.table {
          font-size: smaller;
        }

        pre, table.table {
            font-size: smaller; 
        }
        """
        )
    ),
    ui.row(
        ui.h2("Relationship between Credit Limit and Other Features"),
    ),
    ui.row(
        ui.column(
            4,
            ui.input_select("feature", "Feature:", choices=keep_cols),
            
            ),
        
        ui.column(
            8,
            ui.output_plot("plot1", click=True, dblclick=True, hover=True, brush=True),ui.column(12, ui.output_text_verbatim("click_info")),
            ui.column(12, ui.output_text_verbatim("hover_info")),
            
    ),
        ),
    ui.row(
        ui.h2("Cluster Decision"),
    ),
    ui.row(
        ui.column(6,
                  ui.h4("Silohuette Score"),
                  ui.output_plot("silohuette_score")
                  ),
        ui.column(6,
                  ui.h4("Inertia Score"),
                  ui.output_plot("inertia_score")
                  )
    ),
    ui.row(ui.output_text_verbatim("cluster_decision"),),
    ui.row(
        ui.h2("Decide given data in which cluster"),
    ),
    ui.row(ui.column(12,
                     ui.h4("Feature Importance- Logistic Regression Classifer for Different Classes"),
                     ui.output_plot("feature_important"),
    )),
    ui.row(ui.output_text_verbatim("feature_importance"),),
    ui.row(
        ui.h2("Predict Cluster and Credit_limit of the New Data Points"),
    ),
    ui.row(
        ui.column(4, 
                     ui.panel_well(
                ui.input_radio_buttons(
                    "plot_type1", "Predicted Credit Limit for Different Cluster", ["Credit limit for Cluster 0", "Credit limit for Cluster 1",
                                                                                  "Credit limit for Cluster 2", "Credit limit for Cluster 3",
                                                                                  "Credit limit for Cluster 5","Credit limit for Cluster 6",
                                                                                  "Credit limit for Cluster 7", "Credit limit for Cluster 8"]
                )
            ),
        ),
        
        ui.column(
            8,
            ui.output_plot("plot2",click=True, dblclick=True, hover=True, brush=True),
            
    ),
        ),
    ui.row(ui.output_text_verbatim("cluster"),),
    
)
    
def server(input, output, session):
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
        selected_feature=input.feature()
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

    @output
    @render.plot(alt="plot")
    def silohuette_score(): #Histogram of credit limit from train data
        clusters = range(min_clusters, max_clusters+1)
        fig, ax = plt.subplots(1,2, figsize=(16,6))
        ax.flatten()
        sns.lineplot(x = clusters, y=silohuette_metric, marker='x', linewidth=2, color='b', ax=ax[0])
        sns.lineplot(x = clusters, y=innertia_metric, marker='x', linewidth=2, color='b', ax=ax[1])
        ax[0].axvline(x=clusters[np.argmax(silohuette_metric)], color='red', linestyle='--', label='Best number of clusters')
        ax[1].axvline(x=9, color='red', linestyle='--', label='Best number of clusters',)
        plt.xlabel('Number of Clusters')
        ax[0].set(ylabel='Silohuette Score')
        ax[1].set(ylabel='Inertia Score')
        plt.suptitle('Score Across Different Numbers of Clusters')
        
        return plt.gcf()
    @output
    @render.plot(alt="plot")
    def inertia_score():
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        # Using the credit limit as feature for the Kmeans as it is important to take into account for segmentation.
        scaler = StandardScaler()
        X_data = scaler.fit_transform(df_train[keep_cols])
        min_clusters = 3
        max_clusters = 30
        innertia_metric =[]
        for k in range(min_clusters, max_clusters + 1):
            labels = Kmeans.predict(X_data)
            innertia_metric.append(Kmeans.inertia_)
        clusters = range(min_clusters, max_clusters+1)
        fig, ax = plt.subplots(figsize=(16,6))
        sns.lineplot(x = clusters, y=innertia_metric, marker='x', linewidth=2, color='b')
        ax.axvline(x=9, color='red', linestyle='--', label='Best number of clusters',)
        # Set labels and title
        plt.xlabel('Number of Clusters')
        ax.set(ylabel='Inertia Score')
        plt.suptitle('Score Across Different Numbers of Clusters')
        
        return fig
    
    @output
    @render.plot(alt="feature plot")
    def feature_important():
        df = df_train
        best_n = 9
        scaler = StandardScaler()
        X_data = scaler.fit_transform(df[keep_cols])
        df['kmeans_9'] = Kmeans(n_clusters=9, n_init='auto').predict(X_data)

    # transformers
        numeric_transf = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    
    # Setup the preprocessing steps
        preprocess = ColumnTransformer(transformers=[('num', numeric_transf, selector(dtype_exclude='object'))])

    # Create variables by feature engineering
        df['purch_per_trx'] = np.where(df['PURCHASES_TRX'] > 0, df['PURCHASES'] / df['PURCHASES_TRX'], 0)
        df['cash_adv_per_trx'] = np.where(df['CASH_ADVANCE_TRX'] > 0, df['CASH_ADVANCE'] / df['CASH_ADVANCE_TRX'], 0)
        df['BALANCE_by_freq'] = df['BALANCE'] * df['BALANCE_FREQUENCY']
        df['ONEOFF_PURCHASES_by_freq'] = df['ONEOFF_PURCHASES'] * df['ONEOFF_PURCHASES_FREQUENCY']
        df['INSTALLMENTS_PURCHASES_by_freq'] = df['INSTALLMENTS_PURCHASES'] * df[
        'PURCHASES_INSTALLMENTS_FREQUENCY']
        df['CASH_ADVANCE_by_freq'] = df['CASH_ADVANCE'] * df['CASH_ADVANCE_FREQUENCY']
        df['interest_p_year'] = np.where(df['TENURE'] > 0,
                                     (df['BALANCE'] - ((df['PURCHASES'] + df['CASH_ADVANCE']) -
                                                       (df['PAYMENTS'] + df['MINIMUM_PAYMENTS']))) / df['TENURE'],
                                     (df['BALANCE'] - (df['PURCHASES'] + df['CASH_ADVANCE']) -
                                      (df['PAYMENTS'] + df['MINIMUM_PAYMENTS'])))

        X = df[keep_cols1]
    # The task will be to classify the point to a cluster
        y = df['kmeans_9']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
        y_pred = mk.predict(X_test)

        data = {'Feature': np.tile(keep_cols1, best_n), 'Class': np.repeat(range(best_n), len(keep_cols1)),
            'Coefficient': np.concatenate(mk.named_steps['logit'].coef_)}

        df_plot = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(12, 20))
        bar_width = 0.8

        sns.barplot(data=df_plot, x='Feature', y='Coefficient', hue='Class', ax=ax)
        # Ensure all classes are included in legend labels
        ax.set_title('Feature Importance - Logistic Regression Classifier for Different Classes')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Coefficient Value')
        ax.tick_params(axis='x', rotation=45)
        # Adjust x-axis label size
        ax.set_xticklabels(ax.get_xticklabels(), size=7)
        ax.set_yticklabels(ax.get_yticklabels(), size=7)
        # Move legend outside the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1),title_fontsize='10', fontsize='8')
        plt.tight_layout()
        return fig

    @output
    @render.plot(alt="cluster")
    def plot2(): #plot that shows regression line vs real data
        df = pd.read_csv(Path(__file__).parent / "new_data_with_credit_limit.csv")
        df_tr = pd.read_csv(Path(__file__).parent /'CreditCardUsage.csv')
        target = 'CREDIT_LIMIT'
        # Assuming 'CUST_ID' is the common column in both DataFrames
        common_column = 'CUST_ID'
# Merge the DataFrames based on 'CUST_ID'
        df = pd.merge(df, df_tr[['CUST_ID', 'CREDIT_LIMIT']], on=common_column, how='inner')
        if input.plot_type1() == "Credit limit for Cluster 0":           
            y_test_cluster = df[df['cluster'] == 0][target]
            y_pred = df[df['cluster'] == 0]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {0}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
            #return model_results(y_test, y_pred, f'GB Feat Engin Cluster {0}')
        elif input.plot_type1() == "Credit limit for Cluster 1":
            y_test_cluster = df[df['cluster'] == 1][target]
            y_pred = df[df['cluster'] == 1]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {1}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
        elif input.plot_type1() == "Credit limit for Cluster 2":
            y_test_cluster = df[df['cluster'] == 2][target]
            y_pred = df[df['cluster'] == 2]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {2}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
            
        elif input.plot_type1() == "Credit limit for Cluster 3":
            y_test_cluster = df[df['cluster'] == 3][target]
            y_pred = df[df['cluster'] == 3]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {3}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
        #elif input.plot_type1() == "Credit limit for Cluster 4":
            #y_test_cluster = df[df['cluster'] == 4][target]
            #y_pred = df[df['cluster'] == 4]['CREDIT_PREDICTED']
            #model = f'GB Feat Engin Cluster {4}'
            #fig, ax = plt.subplots()
            #sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            #plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
            #     color='red', linewidth=2)
            #plt.xlabel('Real Values (y_test)')
            #plt.ylabel('Predicted Values (y_pred)')
            #plt.title(f'Real vs. Predicted Values {model}')
            #return fig
        elif input.plot_type1() == "Credit limit for Cluster 5":
            y_test_cluster = df[df['cluster'] == 5][target]
            y_pred = df[df['cluster'] == 5]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {5}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
        elif input.plot_type1() == "Credit limit for Cluster 6":
            y_test_cluster = df[df['cluster'] == 6][target]
            y_pred = df[df['cluster'] == 6]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {6}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
        elif input.plot_type1() == "Credit limit for Cluster 7":
            y_test_cluster = df[df['cluster'] == 7][target]
            y_pred = df[df['cluster'] == 7]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {7}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
        elif input.plot_type1() == "Credit limit for Cluster 8":
            y_test_cluster = df[df['cluster'] == 8][target]
            y_pred = df[df['cluster'] == 8]['CREDIT_PREDICTED']
            model = f'GB Feat Engin Cluster {8}'
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test_cluster, y=y_pred, color='blue', alpha=0.7)
            plt.plot([min(y_test_cluster), max(y_test_cluster)], [min(y_test_cluster), max(y_test_cluster)], linestyle='--',
                 color='red', linewidth=2)
            plt.xlabel('Real Values (y_test)')
            plt.ylabel('Predicted Values (y_pred)')
            plt.title(f'Real vs. Predicted Values {model}')
            return fig
app = App(app_ui, server, debug=True)

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency 
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix, classification_report

#******************************************load csv file**********************************
path_activity_context_tracking = "dataset/activity_context_tracking_data.csv"
path_activity_context_tracking_clean = "dataset/activity_context_tracking_data_clean.csv"


def get_main_data():
    try:
        return pd.read_csv(path_activity_context_tracking)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {path_activity_context_tracking}")
        return None

def get_clean_data():
    try:
        return pd.read_csv(path_activity_context_tracking_clean)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {path_activity_context_tracking_clean}")
        return None

#******************************************load csv file**********************************


#*******************************************Start EDA*******************************************
    
    
def pi_chart():
    try:
        data = get_main_data()
    except:
        print("Error: Failed to retrieve data")
        return

    try:
        plt.figure(figsize=(8, 6))  
        activity_counts = data['activity'].value_counts()
        wedges, texts, autotexts = plt.pie(activity_counts, labels=None, autopct='%1.1f%%')
        plt.title("Pie Chart Activity")
        plt.yticks(rotation=90)
        plt.yticks(fontsize=12)

        legend_labels = [f'{label}: {size.get_text()}' for label, size in zip(activity_counts.index, autotexts)]
        plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Activity Legend")
        plt.savefig('pi.png', dpi=300, bbox_inches='tight')
        plt.show()
    except:
        print("Error: Failed to create pie chart")
        
    
    
def Activity_Frequency():
    try:
        data = get_main_data()
        activity_counts = data["activity"].value_counts()
        activity_counts.plot(kind="bar")
        plt.title("Activity Frequency")
        plt.xlabel("Activity")
        plt.ylabel("Frequency")
        #plt.savefig('eda22.png', dpi=300, bbox_inches='tight') 
        plt.show()
        print(activity_counts)
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"Error: {e}")  
        

def Histograms_of_Each_Column_with_Activity():
    data =  get_clean_data()

    columns = ['orX', 'orY', 'orZ', 'rX', 'rY', 'rZ', 'accX', 'accY', 'accZ', 'gX', 'gY', 'gZ', 'mX', 'mY', 'mZ','lux','soundLevel']

    grouped_data = data.groupby('activity')
    num_rows = int(np.ceil(len(columns) / 3))
    num_cols = min(3, len(columns))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axs = axs.ravel()  

    for i, column in enumerate(columns):
        for activity, group in grouped_data:
            axs[i].hist(group[column], bins=10, alpha=0.5, label=activity)

        axs[i].set_title(column)

    fig.text(0.5, 0.04, 'Values', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.suptitle('Histograms of Each Column with Activity ')
    plt.tight_layout()
    plt.savefig('Histograms.png', dpi=300, bbox_inches='tight')

    plt.show()
        
        
        
def Correlation_Matrix_Heatmap():
    try:
        df =  get_main_data()
        columns_for_correlation = ['orX', 'orY', 'orZ', 'rX', 'rY', 'rZ', 'accX', 'accY', 'accZ', 'gX', 'gY', 'gZ', 'mX', 'mY', 'mZ', 'lux', 'soundLevel', 'activity']

        correlation_matrix = df[columns_for_correlation].corr()
        print(correlation_matrix)

        plt.figure(figsize=(12, 10))  
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

        plt.title('Correlation Matrix Heatmap')
        #plt.savefig('Correlation Matrix Heatmap.png', dpi=300, bbox_inches='tight') 
        plt.show()
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"Error: {e}")
    
   
    
def eda_descriptive_statistical_analysis1():
    try:
        data = get_main_data()

        column = ["orX","orY","orZ","rX","rY","rZ","accX","accY","accZ","gX","gY","gZ","mX","mY","mZ","lux", "soundLevel"]
    
    
        i= 0 ;
    
        for x in column :
            col = column[i]
            i=i+1 
        
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
            mean = np.mean(data[col])
            median = np.median(data[col])
            std_dev = np.std(data[col])
            variance = np.var(data[col])
            min_val = np.min(data[col])
            max_val = np.max(data[col])
            skewness = pd.Series(data[col]).skew()
            kurtosis = pd.Series(data[col]).kurtosis()

            #print results
            print("********************",col,"*********************")
            print("column: ",col," Mean:", mean)
            print("column: ",col," Median:", median)
            print("column: ",col," Standard deviation:", std_dev)
            print("column: ",col," Variance:", variance)
            print("column: ",col," Minimum:", min_val)
            print("column: ",col," Maximum:", max_val)
            print("column: ",col," Skewness:", skewness)
            print("column: ",col," Kurtosis:", kurtosis)
            print("*************************************************")
    except Exception as e:
        print("An error occurred: ", e)
    



   
    
def create_Average_Magnetic_by_Activity():
    try:
        data = get_main_data()
        magnetic_means = data.groupby("activity")[["mX", "mY", "mZ"]].mean()
        magnetic_means.plot(kind="bar")
        plt.title("Average Magnetic by Activity")
        plt.xlabel("Activity")
        plt.ylabel("Magnetic")
        #print(magnetic_means)
        #plt.savefig('create_Average_Magnetic_by_Activity.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Error occurred while creating the Average Magnetic by Activity plot:", str(e))
    
    

    
def create_Average_Gyroscope_by_Activity():
    try:
        data = get_main_data()
        gyroscope_means = data.groupby("activity")[["gX", "gY", "gZ"]].mean()
        gyroscope_means.plot(kind="bar")
        plt.title("Average Gyroscope by Activity")
        plt.xlabel("Activity")
        plt.ylabel("Gyroscope")
        #print(gyroscope_means)
        #plt.savefig('create_Average_Gyroscope_by_Activity.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error occurred: {e}")
    
    

    
def create_Average_Accelerometer_by_Activity():
    try:
        data = get_main_data()
        accelerometer_means = data.groupby("activity")[["accX", "accY", "accZ"]].mean()
        accelerometer_means.plot(kind="bar")
        plt.title("Average Accelerometer by Activity")
        plt.xlabel("Activity")
        plt.ylabel("Accelerometer")
        #print(accelerometer_means)
        #plt.savefig('create_Average_Accelerometer_by_Activity.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Error:", e)
    

    
    
    
    
def create_Average_Rotation_by_Activity():
    try:
        data = get_main_data()
        rotation_means = data.groupby("activity")[["rX", "rY", "rZ"]].mean()
        rotation_means.plot(kind="bar")
        plt.title("Average Rotation by Activity")
        plt.xlabel("Activity")
        plt.ylabel("Rotation")
        #print(rotation_means)
        #plt.savefig('create_Average_Rotation_by_Activity.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("An error occurred:", e)
    
    
    
    
    
def create_Average_Orientation_by_Activity():
    try:
        data = get_main_data()
        orientation_means = data.groupby("activity")[["orX", "orY", "orZ"]].mean()
        orientation_means.plot(kind="bar")
        plt.title("Average Orientation by Activity")
        plt.xlabel("Activity")
        plt.ylabel("Orientation")
        #print(orientation_means)
        #plt.savefig('create_Average_Orientation_by_Activity.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
       


        
def check_balance_data():
    try:
        df =  get_main_data()

        df['activity'].hist()
        plt.title('check balance data')
        plt.xlabel('Activity')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.show()

        class_freq = df['activity'].value_counts()
        print(class_freq)

        imbalance_ratio = class_freq.min() / class_freq.max()
        print('Imbalance ratio:', imbalance_ratio)

        obs = pd.crosstab(df['activity'], columns='count')
        chi2, p, dof, expected = chi2_contingency(obs)
        print('p-value:', p)

        X = df.drop('activity', axis=1)
        y = df['activity']

        skf = StratifiedKFold(n_splits=5)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
    
        print('Cross-validation scores:', scores)
    except Exception as e:
        print('Error:', e)
        
        
        
def balance_data_with_smote():
    try:
        df =  get_clean_data()

        X = df.drop('activity', axis=1) 
        y = df['activity']

        print("Before balance : ")
        print(df['activity'].value_counts())
        print("")

        smote = SMOTE()

        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("After balance : ")
        print(y_balanced.value_counts())

    except Exception as e:
        print('Error:', e)
    

#*******************************************End EDA*******************************************

#*********************************Start Machine Learning***************************************
def clean_data():
    try:
        print("Start clean data")
        data = get_main_data()
        #drop the Additional columns
        data = data.drop(['_id'], axis=1)
        #Remove duplicate rows
        data = data.drop_duplicates()
        # Remove missing values
        data = data.dropna()
        #Save file after clean
        data.to_csv(path_activity_context_tracking_clean, index=False)
        print("finish clean data")
    except FileNotFoundError:
        print(f"Error: File {path_activity_context_tracking} not found.")
    except Exception as e:
        print(f"Error: {e}")
    

    
    
     
    #*********************************End Machine Learning***************************************
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier


df = pd.read_csv(r"c:\Users\user\Downloads\archive(1)\house_price_regression_dataset.csv")

#removes any row with NA data
def drop_na_rows(df:pd.DataFrame):
    df = df.copy()
    df.dropna(inplace=True)

    return df 

#Computes correlation matrix and return the matrix and a heatmap 
def correlation_analysis(df:pd.DataFrame,heatmap_color:str):
    try:
        fig,ax = plt.subplots(1,1)
        colors = ["rocket","mako","flare","crest"]
        df= df.copy()
        corr = df.corr()
        if heatmap_color not in colors :
            return {f"Heatmap Color must be in{colors}"}
        sns.heatmap(corr,cmap = heatmap_color,ax=ax)
        
        return (corr, fig)
    except Exception as e:
        return {f"Error while computing correlation anaylsis :{e}"}
    
    
    
    

# Computes Feature importance and returns feature importnace, bar plot , test score of the rf    
def feature_importance(df:pd.DataFrame,class_label:str,test_size :int):
    if df[class_label].dtypes != "string":
        rf = RandomForestClassifier(random_state=42)
    else:
        rf = RandomForestRegressor(random_state=42)
        
    df = df.copy()
    columns = df.columns.to_list()
    label = df[class_label]
    x_columns = columns.remove(class_label)
        
    x = df.loc[ : , df.columns != class_label]

    X = x.to_numpy()
    Y = label.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size, random_state=42)

    try:
        rf.fit(X_train,y_train)
        test_score = rf.score(X_test,y_test)
        importance = rf.feature_importances_
        importance_series = pd.Series(importance,index=x_columns)
        _, ax = plt.subplots()
        plot = importance_series.plot.bar(yerr=importance, ax=ax)
        
        return (importance, plot ,test_score)
        
    except Exception as e:
        return {f"Error while computing feature importance :{e}"}
    
    
        
test_1 = drop_na_rows(df=df)
corr,_= correlation_analysis(df=df,heatmap_color="crest")
imp = feature_importance(df,class_label="House_Price",test_size=0.2)

print(f"removing null rows:\n {df.shape} after removing {test_1.shape}")
print("-"*50)
print(f"heatmap:{_.show()}")
print("-"*50)
print(f"feature_importance :{imp}")
print(test_1["House_Price"].dtypes)
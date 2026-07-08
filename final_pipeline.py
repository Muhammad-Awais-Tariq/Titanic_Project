import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , FunctionTransformer , StandardScaler , OrdinalEncoder
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier

titanic_df = pd.read_csv("F://Titanic_Project//Data//train (2).csv")

X = titanic_df.drop(columns=["Survived"])
y = titanic_df["Survived"]

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.30 , random_state=42)

age_median = x_train["Age"].median()
embarked_mode = x_train["Embarked"].mode()[0]
fare_median = x_train["Fare"].median()
fare_bins = x_train["Fare"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values.copy()
fare_bins[0] = -np.inf   
fare_bins[-1] = np.inf  

def transform_data(df):
    X = df.copy()
    X["Age"] = X["Age"].fillna(age_median)
    X["Embarked"] = X["Embarked"].fillna(embarked_mode)
    X["Family"] = X["SibSp"] + X["Parch"] + 1
    X["Is_married"] = X["Name"].str.contains("Mrs", na=False).astype(int)
    X["Family_size"] = np.where(X["Family"] >= 4, "Large", "Small")
    X["Is_alone"] = np.where(X["Family"] == 1, 1, 0)
    X["Age_bracket"] = pd.cut(X['Age'], bins=[-np.inf, 12, 18, 60, np.inf], labels=['child','teen','adult','senior'])
    X["Has_cabin"] = X["Cabin"].notna().astype(int)
    X["Fare"] = X["Fare"].fillna(fare_median)
    X["Fare_bracket"] = pd.cut(X["Fare"], bins=fare_bins, labels=["Low_fare","Medium_fare","High_fare","Very_high_fare"])
    X.drop(columns=["PassengerId","Name","Age","SibSp","Parch","Ticket","Fare","Cabin","Family"], inplace=True)
    return X

numeric_coloumns = ["Pclass" , "Is_married" , "Is_alone" , "Has_cabin"]
categorical_coloumns_one_hot = ["Sex" , "Embarked"]
categorical_coloumns_ordinal = ["Family_size", "Age_bracket", "Fare_bracket"]

num_pipeline = Pipeline([
    ("Scaler" , StandardScaler())
])

categorical_one_hot_pipeline = Pipeline([
    ("Encoder" , OneHotEncoder(handle_unknown="ignore"))
])

categorical_ordinal_pipeline = Pipeline([
    ("Encoder" , OrdinalEncoder(
        categories=[
            ["Small" , "Large"],
            ['child','teen','adult','senior'],
            ["Low_fare" , "Medium_fare" , "High_fare" , "Very_high_fare"],
        ]
    ))
])

preprocessor = ColumnTransformer([
    ("num" , num_pipeline , numeric_coloumns),
    ("cat_one" , categorical_one_hot_pipeline , categorical_coloumns_one_hot),
    ("cat_ord" , categorical_ordinal_pipeline , categorical_coloumns_ordinal)
])

random_forest_pipeline = Pipeline([
    ("Feature_enginnering" , FunctionTransformer(transform_data)),
    ("Preprocessing" , preprocessor),
    ("Model" , RandomForestClassifier(n_estimators=50 , min_samples_split=20 , min_samples_leaf=1,max_depth=10 , max_features="log2", random_state=42 , criterion="gini" ) )
])

random_forest_pipeline.fit(x_train, y_train)

final_test = pd.read_csv("F://Titanic_Project//Data//test (2).csv")
passenger_ids = final_test["PassengerId"]
prediction = random_forest_pipeline.predict(final_test)

pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived" : prediction
}).to_csv("F://Titanic_Project//Finalprediction.csv" , index=False)
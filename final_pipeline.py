import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , FunctionTransformer , StandardScaler , OrdinalEncoder
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from transformer import TitanicTransformer
import joblib

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

transform_data = TitanicTransformer(age_median, embarked_mode, fare_median, fare_bins)

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

Tree_pipline = Pipeline([
    ("Feature_enginnering" , FunctionTransformer(transform_data)),
    ("Preprocessing" , preprocessor),
    ("Model" , DecisionTreeClassifier(max_depth=3 , criterion="gini" , min_samples_split=15 , random_state=42 , min_samples_leaf=5) )
])

Tree_pipline .fit(x_train, y_train)

final_test = pd.read_csv("F://Titanic_Project//Data//test (2).csv")
passenger_ids = final_test["PassengerId"]
prediction = Tree_pipline .predict(final_test)

pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived" : prediction
}).to_csv("F://Titanic_Project//Finalprediction.csv" , index=False)

joblib.dump(Tree_pipline, "F://Titanic_Project//tree_pipeline.joblib")
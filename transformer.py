import pandas as pd
import numpy as np

class TitanicTransformer:
    def __init__(self, age_median, embarked_mode, fare_median, fare_bins):
        self.age_median = age_median
        self.embarked_mode = embarked_mode
        self.fare_median = fare_median
        self.fare_bins = fare_bins

    def __call__(self, df):
        X = df.copy()
        X["Age"] = X["Age"].fillna(self.age_median)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode)
        X["Family"] = X["SibSp"] + X["Parch"] + 1
        X["Is_married"] = X["Name"].str.contains("Mrs", na=False).astype(int)
        X["Family_size"] = np.where(X["Family"] >= 4, "Large", "Small")
        X["Is_alone"] = np.where(X["Family"] == 1, 1, 0)
        X["Age_bracket"] = pd.cut(X['Age'], bins=[-np.inf, 12, 18, 60, np.inf], labels=['child','teen','adult','senior'])
        X["Has_cabin"] = X["Cabin"].notna().astype(int)
        X["Fare"] = X["Fare"].fillna(self.fare_median)
        X["Fare_bracket"] = pd.cut(X["Fare"], bins=self.fare_bins, labels=["Low_fare","Medium_fare","High_fare","Very_high_fare"])
        X.drop(columns=["PassengerId","Name","Age","SibSp","Parch","Ticket","Fare","Cabin","Family"], inplace=True)
        return X
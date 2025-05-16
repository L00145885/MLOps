from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = read_csv("LoanApprovalPrediction.csv")

df.drop(['Loan_ID'],axis=1,inplace=True)

X = data.drop(['Loan_Status'],axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

dump(model, "LoanModel.pkl")
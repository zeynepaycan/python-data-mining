import pandas as pd
from sklearn.tree import DecisionTreeRegressor

file_path = 'UK.csv'
UK_data = pd.read_csv(file_path)
print(UK_data.describe())
print(UK_data.columns)
print(UK_data.head(10))
print(UK_data['NickClegg'].max())
print(UK_data['DavidCameron'].max())
print(UK_data['EdMiliband'].max())
print(UK_data['NickClegg'].mean())
print(UK_data['DavidCameron'].mean())
print(UK_data['EdMiliband'].mean())
print(UK_data['NickClegg'].min())
print(UK_data['DavidCameron'].min())
print(UK_data['EdMiliband'].min())

UK_minister = ['DavidCameron','EdMiliband', 'NickClegg']
UK_datas = UK_data[UK_minister]
UK_model = DecisionTreeRegressor(random_state=1)
UK_model.ft(UK_datas,y)

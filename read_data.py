import pandas as pd

patient_info = pd.read_csv('/Users/hongdnn/Downloads/Disease_Forcast.csv')

info = patient_info.sort_values(by=['year', 'month', 'city'])

illness_count = patient_info.groupby(['city', 'illness']).size().reset_index(name='patient_count')

print(info, '\n')

print(illness_count)



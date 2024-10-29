import pandas as pd


data = pd.read_csv('./dataset/Patient_List_2024.csv')

# Group by city, illness, and month, then count occurrences
grouped = data.groupby(['city', 'illness', 'month']).size().reset_index(name='count')

pivot_df = grouped.pivot_table(index=['city', 'illness'], columns='month', values='count', fill_value=0).astype(int)

# Save latest month column which has value for each row
def propagate_values(row):
    latest_month_patients = 0
    latest_month = 0
    # get the number of patient
    for month in range(12, 0, -1):
        if month in row and row[month] > 0:
            print(month, row[month])
            latest_month_patients = row[month]
            latest_month = month
            break

    for month in range(1, 13):
        # Update the number of patients in the months after the latest_month
        if month not in row:
            if month > latest_month:
                row[month] = latest_month_patients
            else:
                row[month] = 0
        elif month > latest_month:
            row[month] = latest_month_patients

    return row

# Apply this function row-wise to pivot_df
pivot_df = pivot_df.apply(propagate_values, axis=1)

# Reorder columns to have months in ascending order
pivot_df = pivot_df[sorted(pivot_df.columns)]

# Reset index to flatten the table
pivot_df = pivot_df.reset_index()
pivot_df.columns.name = None
pivot_df.columns = pivot_df.columns.map(str)

# Display the resulting DataFrame
print(pivot_df)

pivot_df.to_csv('./dataset/Disease_Forcast_2024.csv', index=False)


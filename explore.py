import pandas as pd
df = pd.read_csv('Student Depression Dataset.csv')

with open('data_info.txt', 'w') as f:
    f.write("Columns and Types:\n")
    f.write(str(df.dtypes) + "\n\n")
    
    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")

    f.write("Sample Data (First 3 rows):\n")
    pd.set_option('display.max_columns', None)
    f.write(str(df.head(3)) + "\n\n")
    
    f.write("Description (Summary):\n")
    f.write(str(df.describe(include='all')) + "\n")

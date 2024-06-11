from scipy.io import arff
import pandas as pd

data = arff.loadarff('TimeBasedFeatures-Dataset-15s-NO-VPN.arff')
df = pd.DataFrame(data[0])
df.head()
pd.set_option('display.max_columns', None)

print(df)
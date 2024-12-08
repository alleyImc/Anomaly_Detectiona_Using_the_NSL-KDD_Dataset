from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
data, meta = arff.loadarff("datasets/KDDTrain+.arff")
df = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
print(df.head())
print(df.columns)

# show Protocol Type Distribution
sns.countplot(x=df['protocol_type'].str.decode('utf-8'))
plt.title('Protocol Type Distribution')
plt.show()

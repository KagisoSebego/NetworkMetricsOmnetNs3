import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
size = 50000

enhanced_data_throughput = {
    "Technology": np.random.choice(['2G', '3G', '4G', '5G'], size),
    "Network Load (%)": np.random.uniform(0, 100, size),
    "Bandwidth Allocation (Mbps)": np.random.exponential(scale=1, size=size),
    "Signal Quality (dBm)": np.random.normal(loc=-85, scale=10, size=size),
    "Network Reliability (%)": np.random.uniform(80, 100, size),
    "External Interferences (units)": np.random.normal(loc=0, scale=0.5, size=size),
    "Environmental Factors": np.random.choice(["Clear", "Rainy", "Windy", "Foggy"], size),
    "Data Throughput (Mbps)": np.random.exponential(scale=20, size=size)
}

df_enhanced_data_throughput = pd.DataFrame(enhanced_data_throughput)

numeric_df = df_enhanced_data_throughput.select_dtypes(include=[np.number])  # Keeps only numeric columns
correlation_matrix = numeric_df.corr()

sns.scatterplot(x='Network Load (%)', y='Data Throughput (Mbps)', data=df_enhanced_data_throughput)
plt.title('Network Load vs Data Throughput')
plt.show()

sns.histplot(df_enhanced_data_throughput['Data Throughput (Mbps)'], kde=True)
plt.title('Distribution of Data Throughput')
plt.show()

sns.boxplot(x='Technology', y='Data Throughput (Mbps)', data=df_enhanced_data_throughput)
plt.title('Data Throughput by Technology')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Data')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix with Encoded Data')
plt.show()

sns.pairplot(df_enhanced_data_throughput[['Network Load (%)', 'Signal Quality (dBm)', 'Data Throughput (Mbps)']])
plt.show()

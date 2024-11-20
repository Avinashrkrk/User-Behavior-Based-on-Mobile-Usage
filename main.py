
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('user_behavior_dataset.csv')

df.head()

df.describe()

df.isnull().sum()

df.info()

device_mapping = {
    'Xiaomi Mi 11': 0,
    'iPhone 12': 1,
    'Google Pixel 5': 2,
    'OnePlus 9': 3,
    'Samsung Galaxy S21': 4
}

os_mapping = {
    'Android': 1,
    'iOS': 0
}

gender_mapping = {
    'Male': 1,
    'Female': 0
}

df['Device Model'] = df['Device Model'].map(device_mapping)
df['Operating System'] = df['Operating System'].map(os_mapping)
df['Gender'] = df['Gender'].map(gender_mapping)

df.head(

)

sns.histplot(df['Data Usage (MB/day)'])

plt.figure(figsize=(20, 15))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    linecolor='black',
    cbar_kws={'shrink': 0.8}
)

plt.title('Correlation Heatmap', fontsize=20, pad=20)

plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(
    df['Number of Apps Installed'],
    df['Battery Drain (mAh/day)'],
    color='dodgerblue',
    alpha=0.7,
    edgecolor='black'
)

plt.xlabel('Number of Apps Installed', fontsize=14)
plt.ylabel('Battery Drain (mAh/day)', fontsize=14)

plt.xticks(np.arange(0, 105, 5), fontsize=12)
plt.yticks(np.arange(0, 4200, 200), fontsize=12)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.title('Scatter Plot: Apps Installed vs. Battery Drain', fontsize=16, pad=15)

plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(
    df['Number of Apps Installed'],
    df['Data Usage (MB/day)'],
    color='seagreen',
    alpha=0.7,
    edgecolor='black'
)

plt.xlabel('Number of Apps Installed', fontsize=14)
plt.ylabel('Data Usage (MB/day)', fontsize=14)

plt.xticks(np.arange(0, 105, 5), fontsize=12)
plt.yticks(np.arange(0, 3200, 200), fontsize=12)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.title('Scatter Plot: Apps Installed vs. Data Usage', fontsize=16, pad=15)

plt.show()

df.columns

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x='Device Model',
    kde=True,
    hue='Operating System',
    multiple='stack',
    palette='Set2',
    edgecolor='black'
)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.xlabel('Device Model', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Distribution of Device Models by Operating System', fontsize=16, pad=15)

plt.legend(title='Operating System', fontsize=12, title_fontsize=14, loc='upper right')

plt.show()

X=df.drop(columns=['User Behavior Class'],axis=1)
y=df['User Behavior Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))


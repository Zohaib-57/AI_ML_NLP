from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4)

df = pd.read_csv('iris_data.csv', header=None)
df.head()
# Encode categorical class labels
class_le = LabelEncoder()
y = class_le.fit_transform(df[4].values)

# Standardize features
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(df.iloc[:, range(0, 4)].values)

# Construct within-class covarinat scatter matrix S_W
S_W = np.zeros((4, 4))
for i in range(3):
    S_W += np.cov(X_train_std[y == i].T)
print(S_W)

# Construct between-class scatter matrix S_B
N = np.bincount(y)
vecs = []
[vecs.append(np.mean(X_train_std[y == i], axis=0))
 for i in range(3)
 ]
means_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((4, 4))
for i in range(3):
    S_B += N[i]*(((vecs[i]-means_overall).reshape(4, 1)).dot(((vecs[i]-means_overall).reshape(1, 4))))
print(S_B)

# Calculate the sorted EigenValues and EigenVectors of inverse (S_B)dot(S_W
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# Project original features onto the new feature space
W = np.hstack((eigen_pairs[0][1][:, ].reshape(4, 1), eigen_pairs[1][1][:, ].reshape(4, 1))).real
X_train_lda = X_train_std.dot(W)

# Plot transformed features in LDA subspace
data = pd.DataFrame(X_train_lda)
data['class'] = y
data.columns = ["LD1", "LD2", "class"]
data.head()


markers = ['s', 'x', 'o']
sns.lmplot(x="LD1", y="LD2", data=data, markers=markers, fit_reg=False, hue='class', legend=False)
plt.legend(loc='upper center')
plt.show()

# LDA implementation using scikit-learn
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y)
data = pd.DataFrame(X_train_lda)
data['class'] = y
data.columns = ["LD1", "LD2", "class"]
data.head()

markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show()

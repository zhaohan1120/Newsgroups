from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 定义停用词表
stop_words = ['a', 'an', 'the', 'and', 'or', 'if', 'is', 'of', 'to', 'this', 'that', 'in', 'it', 'with', 'for', 'not']

# 定义函数去除文本中的数字和符号
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 下载20 Newsgroups数据集
categories = ['comp.graphics', 'comp.os.ms-windows.misc','rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)

# 对训练文档进行预处理和向量化
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, preprocessor=clean_text)
X_tfidf = tfidf_vectorizer.fit_transform(train.data)
X_tfidf_norm = StandardScaler().fit_transform(X_tfidf.toarray())

# 对训练文档进行t-SNE降维，降至2D
tsne_model = TSNE(n_components=2, random_state=0)
X_tsne = tsne_model.fit_transform(X_tfidf_norm)

# 构建K-means模型并进行聚类
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100)
kmeans.fit(X_tfidf_norm)

# 将原始数据和聚类结果转换为DataFrame格式
df = pd.DataFrame(X_tsne, columns=['comp1', 'comp2'])
df['label'] = kmeans.labels_

# 对原始文本进行清洗处理
cleaned_data = list(map(clean_text, train.data))

# 可视化聚类结果
markers = ['o', 'v', '*', 's']

plt.figure(figsize=(10, 8))
for label in range(k):
    plt.scatter(df.loc[df['label']==label, 'comp1'], df.loc[df['label']==label, 'comp2'],
                marker=markers[label], alpha=0.7, label=f'Cluster {label+1}')
    
    for i in np.where(kmeans.labels_==label)[0]:
        plt.annotate(cleaned_data[i][0:20], (df.iloc[i]['comp1'], df.iloc[i]['comp2']))

plt.title('t-SNE Visualization of 20 Newsgroups Dataset')
plt.legend()
plt.show()

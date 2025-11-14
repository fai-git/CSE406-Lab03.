
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("../data/student_data.csv")

# Data Cleaning
df = df.drop_duplicates()
df['attendance'] = df['attendance'].fillna(df['attendance'].mean())
df['marks'] = df['marks'].fillna(df['marks'].mean())
df['attendance'] = df['attendance'].astype(float)
df['marks'] = df['marks'].astype(float)

# Descriptive Statistics
print(df.describe())

# Correlation Analysis
correlation = df[['attendance', 'marks']].corr()
print(correlation)

plt.figure(figsize=(5,4))
plt.imshow(correlation, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Heatmap")
plt.show()

# Performance Trend
plt.figure(figsize=(10,5))
plt.plot(df['marks'], marker='o')
plt.title("Student Marks Trend")
plt.xlabel("Student Index")
plt.ylabel("Marks")
plt.grid()
plt.show()

# Low-Performing Topics
topic_cols = [col for col in df.columns if "topic" in col.lower()]
topic_means = df[topic_cols].mean().sort_values()
print("Average Scores by Topic:")
print(topic_means)

topic_means.plot(kind='bar', figsize=(8,4))
plt.title("Topic-wise Average Performance")
plt.ylabel("Average Marks")
plt.show()

# Outlier Detection
df['zscore_marks'] = (df['marks'] - df['marks'].mean()) / df['marks'].std()
outliers = df[df['zscore_marks'] < -2]
print("Outliers:")
print(outliers)

# Moving Average
df['moving_avg'] = df['marks'].rolling(window=5).mean()
plt.figure(figsize=(10,5))
plt.plot(df['moving_avg'])
plt.title("Smoothed Marks Trend (Moving Average)")
plt.show()

# Recommendation System
def recommend_level(score):
    if score < 40:
        return "Easy"
    elif score < 70:
        return "Medium"
    else:
        return "Advanced"

df['recommendation'] = df['marks'].apply(recommend_level)
print(df[['marks', 'recommendation']].head())

# Summary
print("\nLow Performing Topics:")
print(topic_means.head(3))
print("\nPerformance–Attendance Correlation:")
print(correlation)
print("\nRecommendation Sample:")
print(df[['marks', 'recommendation']].head())

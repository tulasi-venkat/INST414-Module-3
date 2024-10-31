import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import numpy as np

file_path = 'Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv'
data = pd.read_csv(file_path)

data = data[['Survey ID', 'Age', 'Ethnicity', 'Education Completed', 'Household Size']].dropna()

scaler = MinMaxScaler()
data[['Age', 'Household Size', 'Education Completed']] = scaler.fit_transform(
    data[['Age', 'Household Size', 'Education Completed']])

def compute_similarity(query, profile):
    num_features = ['Age', 'Household Size', 'Education Completed']
    num_distance = euclidean(query[num_features], profile[num_features])
    
    similarity = 1 if query['Ethnicity'] == profile['Ethnicity'] else 0
    
    similarity_score = (1 - num_distance) * 0.7 + similarity * 0.3
    return similarity_score

def find_top_similar(data, query_index):
    query = data.iloc[query_index]
    similarities = []
    
    for i, profile in data.iterrows():
        if i != query_index:  
            score = compute_similarity(query, profile)
            similarities.append((profile['Survey ID'], score))
    
    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    return top_similar

queries = [
    {"Age": 0.25, "Household Size": 0.1, "Education Completed": 0.5, "Ethnicity": "Chinese"},  # Query 1
    {"Age": 0.45, "Household Size": 0.4, "Education Completed": 0.3, "Ethnicity": "Vietnamese"},  # Query 2
    {"Age": 0.7, "Household Size": 0.1, "Education Completed": 0.2, "Ethnicity": "Korean"}  # Query 3
]

query_indices = []
for q in queries:
    closest_idx = data.apply(
        lambda row: euclidean([q['Age'], q['Household Size'], q['Education Completed']],
                              [row['Age'], row['Household Size'], row['Education Completed']]), axis=1).idxmin()
    query_indices.append(closest_idx)

for i, query_index in enumerate(query_indices):
    top_similar = find_top_similar(data, query_index)
    print(f"Top 10 most similar profiles to Query {i+1}:")
    for profile_id, score in top_similar:
        print(f"Survey ID: {profile_id}, Similarity Score: {score:.2f}")
    print("\n")

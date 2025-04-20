import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Get the path to the model_folder
model_folder = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained model instead of using model_folder
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load titles and embeddings
with open(os.path.join(model_folder, "titles.json"), "r") as f:
    titles = json.load(f)
title_embeddings = np.load(os.path.join(model_folder, "title_embeddings.npy"))

def find_top_similar_titles(new_title, duplicate_threshold=0.95, similar_threshold=0.80):
    try:
        # Encode the new title
        new_embedding = model.encode([new_title])
        
        # Calculate similarities
        similarities = cosine_similarity(new_embedding, title_embeddings)[0]
        
        # Get top matches
        top_indices = similarities.argsort()[-3:][::-1]
        
        # Convert NumPy values to Python types
        top_titles = [(titles[i], float(similarities[i])) for i in top_indices]
        
        # Determine status
        if top_titles[0][1] > duplicate_threshold:
            status = "Duplicate"
        elif top_titles[0][1] > similar_threshold:
            status = "Similar"
        else:
            status = "Unique"
        
        return {
            "label": status,
            "top_matches": top_titles
        }
    except Exception as e:
        print(f"Error in find_top_similar_titles: {str(e)}")
        raise

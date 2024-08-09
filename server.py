import streamlit as st
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from googletrans import Translator

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    return tokenizer, model

@st.cache_data
def load_embeddings_and_data(file_path):
    with open(file_path, 'rb') as f:
        embeddings, df = pickle.load(f)
    return embeddings, df

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, dest='en')
    return translated.text

def search_products(query, embeddings, df, tokenizer, model, top_n=10000):
    english_query = translate_to_english(query)
    
    encoded_query = tokenizer([english_query], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_embedding = mean_pooling(query_output, encoded_query['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices]
    results['cosine_similarity'] = similarities[top_indices]
    
    return results, english_query

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    st.title("Product Search App")
    
    tokenizer, model = load_model_and_tokenizer()
    embeddings, df = load_embeddings_and_data('product_embeddings.pkl')
    
    df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
    
    df['retail_price'] = pd.to_numeric(df['retail_price'], errors='coerce')
    
    df['brand'] = df['brand'].astype(str)
    
    query = st.text_input("Enter your search query:")
    
    if query:
        results, english_query = search_products(query, embeddings, df, tokenizer, model)
        
        if not results.empty:
            results = results.dropna(subset=['description'])
            
            st.sidebar.header("Filters")
            
            similarity_cutoff = st.sidebar.slider("Cosine similarity cutoff", 0.0, 1.0, 0.2)
            
            filtered_results = results[results['cosine_similarity'] >= similarity_cutoff]
            
            available_brands = sorted(filtered_results['brand'].unique())
            
            selected_brands = st.sidebar.multiselect("Select brands", available_brands)
            
            rating_options = ["All", "Above 4 stars", "Above 3 stars", "Above 2 stars", "Above 1 star"]
            selected_rating = st.sidebar.selectbox("Select rating", rating_options)
            
            valid_prices = filtered_results['retail_price'].dropna()
            if not valid_prices.empty:
                min_price = float(valid_prices.min())
                max_price = float(valid_prices.max())
                if min_price < max_price:
                    price_range = st.sidebar.slider("Price range", min_price, max_price, (min_price, max_price))
                    filtered_results = filtered_results[
                        (filtered_results['retail_price'] >= price_range[0]) & 
                        (filtered_results['retail_price'] <= price_range[1])
                    ]
                else:
                    st.sidebar.text("No valid price range available")
            else:
                st.sidebar.text("No valid prices available")
            
            if selected_brands:
                filtered_results = filtered_results[filtered_results['brand'].isin(selected_brands)]
            
            if selected_rating == "Above 4 stars":
                filtered_results = filtered_results[filtered_results['product_rating'] > 4]
            elif selected_rating == "Above 3 stars":
                filtered_results = filtered_results[filtered_results['product_rating'] > 3]
            elif selected_rating == "Above 2 stars":
                filtered_results = filtered_results[filtered_results['product_rating'] > 2]
            elif selected_rating == "Above 1 star":
                filtered_results = filtered_results[filtered_results['product_rating'] > 1]
            
            st.write(f"Showing {len(filtered_results)} filtered products:")
            
            columns_to_display = ['product_name', 'description', 'retail_price', 'discounted_price', 'brand', 'product_rating', 'cosine_similarity']
            results_table = filtered_results[columns_to_display]
            
            results_table['product_rating'] = results_table['product_rating'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
            results_table['cosine_similarity'] = results_table['cosine_similarity'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(results_table, height=400)
        else:
            st.write("No matching products found.")
    
    st.markdown("---")
    st.markdown("If this app doesn't work, please try [version 1](https://zeptocasestudy-bf3mw3ixbndrax4vfvesns.streamlit.app/).")

if __name__ == "__main__":
    main()
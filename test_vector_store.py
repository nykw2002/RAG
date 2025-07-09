#!/usr/bin/env python3
"""
Test the vector store contents
"""

import pickle
import numpy as np
import os

def test_vector_store():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(script_dir, "vector_store", "local_vector_store.pkl")
    
    print("Testing Vector Store")
    print("=" * 50)
    
    if not os.path.exists(vector_store_path):
        print(f"Vector store not found at: {vector_store_path}")
        return
    
    # Load the vector store
    with open(vector_store_path, 'rb') as f:
        store_data = pickle.load(f)
    
    print("Vector store loaded successfully!")
    print(f"Keys: {list(store_data.keys())}")
    print(f"Number of vectors: {len(store_data['vectors'])}")
    print(f"Number of texts: {len(store_data['texts'])}")
    print(f"Number of metadata entries: {len(store_data['metadata'])}")
    
    if len(store_data['vectors']) > 0:
        vectors = np.array(store_data['vectors'])
        print(f"Vector dimensions: {vectors.shape}")
    
    # Show some sample chunks
    print(f"\nSample chunks:")
    for i in range(min(3, len(store_data['texts']))):
        print(f"\nChunk {i+1}:")
        print(f"Type: {store_data['metadata'][i].get('type', 'unknown')}")
        print(f"Page: {store_data['metadata'][i].get('page_number', 'N/A')}")
        print(f"Text: {store_data['texts'][i][:150]}...")
    
    # Count chunk types
    chunk_types = {}
    for meta in store_data['metadata']:
        chunk_type = meta.get('type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    print(f"\nChunk distribution:")
    for chunk_type, count in chunk_types.items():
        print(f"  {chunk_type}: {count}")

if __name__ == "__main__":
    test_vector_store()

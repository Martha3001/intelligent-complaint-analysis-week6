import ast
import os
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import InferenceClient

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_metadata(metadata_path):
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

def prepare_chunks_and_metadata(df):
    """
    Prepare text chunks and metadata from a DataFrame.
    Splits the text in the specified column into chunks of a given size.
    Returns a list of text chunks and their corresponding metadata.
    """
    all_chunks = []
    metadata = []
    for idx, row in df.iterrows():
        complaint_id = row.get('Complaint ID', idx)  # fallback to index if no ID
        product = row.get('Product', None)
        # Safely evaluate the string representation of the list
        try:
            chunks = ast.literal_eval(row['narrative_chunks'])
        except (ValueError, SyntaxError):
            # Handle cases where the string is not a valid list representation
            chunks = [] # or handle the error as appropriate

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({'complaint_id': complaint_id, 'product': product})

    return all_chunks, metadata

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Load the SentenceTransformer model for embedding text.
    Default is 'all-MiniLM-L6-v2', which is a lightweight model suitable for many tasks.
    """
    return SentenceTransformer(model_name)

def embed_chunks(chunks, embedding_model):
    """
    Embed a list of text chunks using the provided embedding model.
    Returns a numpy array of embeddings.
    """
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings

def initialize_faiss_index(embeddings, index_path='faiss_index.index', metadata_path='metadata.pkl'):
    """
    Initialize a FAISS index with the provided text chunks.
    Saves the index and metadata to specified paths.
    """
    # Create a FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(embeddings)  # Add embeddings to the index
    
    return index

def save_faiss_index(index, metadata, index_path, metadata_path):
    """
    Save the FAISS index to the specified path.
    """
    faiss.write_index(index, index_path)

    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def retrieve_similar_complaints(question, embedding_model, index, metadata_list, all_chunks, k=5):
    """
    Embeds the user's question and retrieves the top-k most similar complaint chunks from the FAISS vector store.
    Returns a list of (chunk, metadata, distance) tuples.
    """
    # Embed the question
    question_embedding = embedding_model.encode([question])[0]
    question_embedding = np.array([question_embedding])
    # Search the FAISS index
    distances, indices = index.search(question_embedding, k)
    # Retrieve results
    results = []
    for i, idx in enumerate(indices[0]):
        chunk_text = all_chunks[idx]
        meta = metadata_list[idx]
        dist = distances[0][i]
        results.append((chunk_text, meta, dist))
    return results

def generate_llm_answer(question, retrieved_chunks, prompt_template, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Combines the prompt, user question, and retrieved chunks, sends to LLM, and returns the generated response.
    """
    # Combine retrieved chunks into context
    context = "\n".join(retrieved_chunks)
    prompt = prompt_template.format(context=context, question=question)
    
    # Load the LLM pipeline (text-generation)
    generator = pipeline("text-generation", model=model_name, max_new_tokens=256)
    
    # Generate response
    response = generator(prompt, return_full_text=False)
    # Extract and return the generated answer
    return response[0]['generated_text'] if response else ""

def generate_llama_llm_answer(question, retrieved_chunks, prompt_template, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key=None):
    """
    Uses Hugging Face InferenceClient with Fireworks provider to generate an answer using Llama-3.
    """
    context = "\n".join(retrieved_chunks)
    prompt = prompt_template.format(context=context, question=question)
    
    # You can pass api_key directly or use environment variable
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=api_key or os.environ.get("HF_TOKEN")
    )
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    # Return the generated message text
    return completion.choices[0].message.content if completion and completion.choices else ""
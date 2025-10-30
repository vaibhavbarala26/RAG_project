import json
import os
from tqdm import tqdm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import get

def preprocessing():
    """
    This function loads data from multiple JSON files, adds a source identifier,
    processes the content into LangChain documents, splits them, and indexes them
    into a single FAISS vector store, saving the final index to disk.
    """
    # 1. Load data from all specified JSON files
    print("Loading JSON data from multiple sources...")
    paths = [
        "./vaibhav/numpy_docs.json",
        "./vaibhav/pandas_docs.json",
        "./vaibhav/scikit__docs.json",
        "./vaibhav/tensor__docs.json",
        "./vaibhav/torch__docs.json",
        "./vaibhav/matplot__docs.json",
        "./vaibhav/lgbm__docs.json",
        "./vaibhav/seaborn__docs.json",
        "./vaibhav/streamlit__docs.json",
        "./vaibhav/transformer__docs.json",
        "./vaibhav/xgb__docs.json"
            ]
    all_data = []
    for path in paths:
        try:
            # ---  FIX 1: Use the 'path' variable to load each file ---
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # --- NEW FEATURE: Add a source identifier to each item ---
                source_name = os.path.basename(path).split('.')[0] # e.g., "pandas_docs"
                for item in data:
                    item['source'] = source_name
                all_data.extend(data)
                print(f"Loaded {len(data)} items from {path}")
        except FileNotFoundError:
            print(f"Warning: File not found at {path}. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {path}. Skipping.")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    print(f"\nTotal items loaded from all sources: {len(all_data)}")

    # 2. Turn all JSON objects into a single flat list of LangChain Documents
    # --- REFACTOR 1: Simplify to a single flat list 'all_docs' ---
    print("Converting JSON objects to LangChain Documents...")
    all_docs = []
    for item in tqdm(all_data, desc="Creating Documents"):
        content = item.get("content", "")
        metadata = {
            "url": item.get("url"),
            "title": item.get("title"),
            "source": item.get("source", "unknown"), # Add the source here
        }
        all_docs.append(Document(page_content=content, metadata=metadata))

    # 3. Split all documents into smaller chunks
    print("Splitting documents into smaller chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = splitter.split_documents(all_docs)
    print(f"Total documents after splitting: {len(chunked_docs)}")

    # 4. Initialize LangChain components
    print("Initializing LLM, Embeddings, and Vector Store...")
    _, embeddings, vector_store = get()

    # 5. Add all document chunks to the vector store in batches
    # --- REFACTOR 2: Simplify to a single, correct indexing loop ---
    print("Adding documents to the vector store...")
    batch_size = 1000
    all_ids = []
    for i in tqdm(range(0, len(chunked_docs), batch_size), desc="Indexing Batches"):
        batch = chunked_docs[i:i + batch_size]
        ids = vector_store.add_documents(documents=batch)
        all_ids.extend(ids)

    print(f"\nSuccessfully added {len(all_ids)} document chunks to the vector store.")

    # 6. Save the final index to disk
    print("Saving FAISS index to disk...")
    vector_store.save_local("faiss_index_multi") # Saved to a new folder
    print("Index saved successfully to the 'faiss_index_multi' folder.")

# --- Main execution block ---
if __name__ == "__main__":
    # IMPORTANT: This will create a new index folder named 'faiss_index_multi'
    preprocessing()

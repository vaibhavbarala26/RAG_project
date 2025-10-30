import os
import textwrap
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from app import get

def setup_components():
    """
    Loads and initializes all the core components needed for the RAG chain,
    but does not assemble the final chain.
    """
    index_path = "faiss_index_multi"
    if not os.path.exists(index_path):
        print(f"Error: FAISS index not found at '{index_path}'.")
        return None, None, None
    
    llm, embeddings, _ = get()
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        llm=llm
    )
    print("FAISS index and core components loaded successfully.")
    return llm, vector_store, memory

# 🔑 Mapping from LLM-friendly names → FAISS metadata source keys
SOURCE_MAP = {
    "pandas": "pandas_docs",
    "numpy": "numpy_docs",
    "scikit-learn": "scikit__docs",
    "sklearn": "scikit__docs",
    "tensorflow": "tensor__docs",
    "torch": "torch__docs",
    "pytorch": "torch__docs",
    "matplotlib":"matplot__docs",
    "seaborn":"seaborn_doocs",
    "streamlit":"streamlit__docs",
    "xgboost":"xgb__docs"
}

# def extract_source_from_query(llm, query, available_sources):
#     """
#     Uses an LLM chain to extract the library source from the user's query,
#     then maps it using SOURCE_MAP.
#     """
#     prompt = PromptTemplate(
#         template="""
#         Analyze the user's question to determine which of the following data science libraries is the primary topic: {sources}.
#         Your response must be ONLY the library names from the provided list.
#         If the question is not about any of these libraries or is ambiguous, respond with the word "None".

#         User Question: "{question}"
#         Library:
#         """,
#         input_variables=["question", "sources"]
#     )
    
#     chain = LLMChain(llm=llm, prompt=prompt)
    
#     source_list_str = ", ".join(SOURCE_MAP.keys())
    
#     result = chain.invoke({"question": query, "sources": source_list_str})
    
#     extracted_source = result['text'].strip().lower()
    
#     if not extracted_source or extracted_source == "none":
#         return None

#     # ✅ Use dictionary lookup instead of loop
#     return SOURCE_MAP.get(extracted_source, None)
import re



def extract_source_from_query(query, source_map=SOURCE_MAP):
    """
    Finds all library names mentioned in a query and returns a list of 
    their corresponding source names.
    """
    query_lower = query.lower()
    found_sources = []
    # Iterate through all possible libraries
    for lib, source in source_map.items():
        # If a library name is found as a whole word in the query...
        if re.search(rf"\b{lib}\b", query_lower):
            # ...add its source name to our list instead of returning.
            found_sources.append(source)
    
    # Return the list of unique sources found, or None if the list is empty.
    return list(set(found_sources)) if found_sources else None



def create_filtered_chain(llm, vector_store, memory, source_filters=None):
    """
    Creates a new ConversationalRetrievalChain with a retriever that is
    dynamically configured with a metadata filter for one or more sources.
    """
    search_kwargs = {"k": 5}
    if source_filters:
        search_kwargs["filter"] = {"source": source_filters}
        print(f"Retriever created with filter: source IN {source_filters}")
    else:
        print("Retriever created with no filter.")
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    prompt_template = """
    You are an expert assistant for Python data science libraries like pandas, NumPy, PyTorch, and Scikit-learn. Your role is to answer the user's question based ONLY on the provided context.
    
    Follow these instructions:
    1. Identify the primary library (e.g., pandas, NumPy) being discussed in the context. Frame your answer from the perspective of an expert in that specific library.
    2. Answer the question directly and concisely.
    3. If the context includes code examples, use them in your answer.
    4. If you don't know the answer from the documentation provided, say "I'm sorry, I don't have enough information from the documentation to answer that question."
    5. At the end of your answer, on a new line, cite the source URL from the metadata.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Helpful Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "chat_history", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def ask_question(chain, question):
    """
    Asks a question using the chain and prints the formatted response.
    """
    result = chain.invoke({"question": question})
    wrapped_answer = textwrap.fill(result["answer"], width=100)
    print(f"\nAnswer:\n{wrapped_answer}")
    
    print("\n--- Sources Used ---")
    if result.get("source_documents"):
        unique_sources = {}
        for doc in result["source_documents"]:
            source = doc.metadata.get('source', 'N/A')
            url = doc.metadata.get('url', 'N/A')
            unique_sources[url] = source
        
        for url, source in unique_sources.items():
            print(f"- Library: {source}, URL: {url}")
    else:
        print("No sources found.")
    print("--------------------\n")


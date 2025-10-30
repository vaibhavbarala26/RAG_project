import os
import textwrap
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from app import get

def setup_conversational_chain():
    """
    Sets up the conversational RAG chain with a dynamic prompt suitable
    for multiple documentation sources.
    """
    index_path = "faiss_index_multi"
    if not os.path.exists(index_path):
        print(f"Error: FAISS index not found at '{index_path}'.")
        print("Please run the multi-file ingestion script first.")
        return None
    
    llm, embeddings, _ = get()
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    print("FAISS index loaded successfully.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 9})
    
    # --- IMPROVED DYNAMIC PROMPT ---
    # This new prompt is more general and instructs the LLM to adapt its persona
    # based on the source of the retrieved documents.
    prompt_template = """
    You are an expert assistant for Python data science libraries like pandas, NumPy, PyTorch, and Scikit-learn. Your role is to answer the user's question based ONLY on the provided context.
    
    Follow these instructions:
    1. Identify the primary library (e.g., pandas, NumPy) being discussed in the context. Frame your answer from the perspective of an expert in that specific library.
    2. Answer the question directly and concisely. Do not start your answer with phrases like "Based on the context...".
    3. If the context includes code examples, use them in your answer.
    4. If you don't know the answer from the context provided, say "I'm sorry, I don't have enough information from the documentation to answer that question."
    5. At the end of your answer, on a new line, cite the source URL from the metadata.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Helpful Answer:
    """
    
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        llm=llm
    )
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "chat_history", "question"]
    )

    print("Creating ConversationalRetrievalChain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    print("Chain is ready!")
    return qa_chain

def ask_question(chain, question):
    """
    Asks a question using the chain and prints the formatted response.
    """
    if not chain:
        return
    
    print(f"\nQuery: {question}")

    result = chain.invoke({"question": question})
    wrapped_answer = textwrap.fill(result["answer"], width=100)
    print(f"\nAnswer:\n{wrapped_answer}")
    
    print("\n--- Sources Used ---")
    if result.get("source_documents"):
        # Create a unique list of sources to avoid repetition
        unique_sources = {}
        for doc in result["source_documents"]:
            source = doc.metadata.get('source', 'N/A')
            url = doc.metadata.get('url', 'N/A')
            unique_sources[url] = source # Use URL as key to ensure uniqueness
        
        for url, source in unique_sources.items():
            print(f"- Library: {source}, URL: {url}")
    else:
        print("No sources found.")
    print("--------------------\n")

if __name__ == "__main__":
    qa_chain = setup_conversational_chain()

    if qa_chain:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            user_question = input("Ask a question: ")
            if user_question.lower() == 'exit':
                print("Exiting chatbot.")
                break
            ask_question(qa_chain, user_question)

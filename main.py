# from load2 import setup_components , SOURCE_MAP , extract_source_from_query , create_filtered_chain , ask_question


# if __name__ == "__main__":
#     llm, vector_store, memory = setup_components()

#     if llm:
#         print("\nEntering interactive mode. Type 'exit' to quit.")
#         available_sources = list(SOURCE_MAP.values())
        
#         while True:
#             user_question = input("Ask a question: ")
#             if user_question.lower() == 'exit':
#                 print("Exiting chatbot.")
#                 break
            
#             # --- AUTOMATIC SOURCE DETECTION ---
#             source_filters = []
#             detected_source = extract_source_from_query(user_question)
            
#             if detected_source:
#                 print(f"--> Automatically detected source: {detected_source}")
#                 source_filters.append(detected_source)
#             else:
#                 # Fallback to manual input if no source is detected
#                 print(f"\nCould not automatically determine the library.")
#                 print(f"Available sources: {', '.join(available_sources)}")
#                 filter_input = input("Enter source(s) to filter by (comma-separated), or press Enter for all: ").strip()

#                 is_valid_input = True
#                 if filter_input:
#                     source_filters = [s.strip() for s in filter_input.split(',')]
#                     for s_filter in source_filters:
#                         if s_filter not in available_sources:
#                             print(f"Invalid source: '{s_filter}'. Please choose from the list.")
#                             is_valid_input = False
#                             break
#                     if not is_valid_input:
#                         continue
            
#             # Create a new chain with the dynamic filter for each question
#             qa_chain = create_filtered_chain(llm, vector_store, memory, source_filters if source_filters else None)
            
#             ask_question(qa_chain, user_question)
# app.py
import streamlit as st
from load2 import setup_components, SOURCE_MAP, extract_source_from_query, create_filtered_chain

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 RAG Chatbot with Source Filtering")
st.caption("A chatbot that can answer questions from multiple document sources.")

# --- 2. LOAD AND CACHE RESOURCES ---
# Use st.cache_resource to load the LLM, vector store, and memory only once.
# This hugely improves the app's performance.
@st.cache_resource
def load_resources():
    """Load and cache the language model, vector store, and memory."""
    print("--- Loading resources ---") # This will print to the console only on the first run
    llm, vector_store, memory = setup_components()
    return llm, vector_store, memory

llm, vector_store, memory = load_resources()
available_sources = list(SOURCE_MAP.values())

# --- 3. SIDEBAR FOR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Allow user to manually select sources. This overrides automatic detection.
    manual_filters = st.multiselect(
        "Manually filter by source(s):",
        options=available_sources,
        help="If you select sources here, the chatbot will only use these. Leave empty for automatic detection."
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 4. INITIALIZE CHAT HISTORY ---
# st.session_state is Streamlit's way of preserving state across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# --- 5. DISPLAY CHAT HISTORY ---
# Iterate through the stored messages and display them.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. HANDLE USER INPUT AND RUN THE CHAIN ---
if prompt := st.chat_input("Ask a question..."):
    # Add user's message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a thinking spinner while processing
    with st.spinner("Thinking..."):
        source_filters = []
        filter_message = ""

        # --- Determine which filters to use ---
        if manual_filters:
            source_filters = manual_filters
            filter_message = f"🔍 **Manually filtering by:** `{', '.join(source_filters)}`"
        else:
            # Automatic source detection
            detected_source = extract_source_from_query(prompt)
            if detected_source:
                source_filters.append(detected_source)
                filter_message = f"🎯 **Automatically detected source:** `{detected_source}`"
            else:
                filter_message = "⚠️ **No specific source detected.** Searching all documents."

        # --- Create and run the QA chain ---
        # A new chain is created for each question with the specific filters.
        qa_chain = create_filtered_chain(llm, vector_store, memory, source_filters if source_filters else None)
        
        # We don't need the separate ask_question function here, we can invoke directly.
        response = qa_chain.invoke({"question": prompt})
        
        # The 'answer' is usually in a specific key in the response dictionary.
        # Adjust 'answer' if your chain returns a different key.
        answer = response.get('answer', "Sorry, I couldn't find an answer.")

        # --- Display the assistant's response ---
        with st.chat_message("assistant"):
            st.markdown(filter_message)
            st.markdown("---")
            st.markdown(answer)
        
        # Add the full assistant response to session state
        full_response = f"{filter_message}\n\n---\n\n{answer}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
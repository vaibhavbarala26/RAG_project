
# 🧠 RAG-based ML Documentation Assistant using Ollama Mistral

A Retrieval-Augmented Generation (RAG) project powered by Ollama Mistral, designed to help developers and data scientists quickly query and understand documentation for popular Machine Learning and Data Science libraries.

This system scrapes official documentation pages of major ML libraries and allows you to ask natural language questions about any concept, function, or usage. It also features memory, so it can remember previous queries and maintain context across conversations.

---

## 🚀 Features

* ⚙️ **RAG Pipeline** — Combines document retrieval with generative responses for accurate and contextual answers
* 🧩 **Ollama Mistral LLM** — Fast and efficient open-weight model for local inference
* 📚 **Documentation Scraping** — Automatically crawls and indexes major ML library docs
* 💬 **Conversational Memory** — Keeps track of past queries for contextual, multi-turn interactions
* 🎛️ **Streamlit Frontend** — Clean and interactive web interface for chatting with the model
* 📈 **Supports ML Libraries:**
   * Streamlit
   * yFinance
   * pandas
   * NumPy
   * scikit-learn
   * TA-Lib
   * TensorFlow
   * Matplotlib
   * Plotly
   * Seaborn

---

## 🏗️ Tech Stack

| Component | Technology Used |
|-----------|----------------|
| **LLM** | Ollama (Mistral Model) |
| **Vector Store** | FAISS |
| **Web App** | Streamlit |
| **Data** | Scraped official documentation pages |
| **ML Libraries Indexed** | pandas, numpy, tensorflow, sklearn, etc. |
| **Memory** | Custom memory module for conversation tracking |

---

## 📊 Architecture Diagram

```
┌─────────────────┐
│  Documentation  │
│     Scraper     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │
│     Store       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│   User Query    │─────▶│   RAG Pipeline  │
│  (Streamlit UI) │      │ (Ollama Mistral)│
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Response with  │
                         │     Context     │
                         └─────────────────┘
```

*This diagram shows the flow of data from the documentation scraper to the vector store (FAISS), which is then queried by the RAG pipeline (using Ollama Mistral) and presented to the user via the Streamlit interface.*

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vaibhavbarala26/RAG_project.git
cd RAG_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows
```

**Additional Instructions:**
* If you are using git-bash on Windows, use the `source venv/bin/activate` command.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Run Ollama

Download and install Ollama from [https://ollama.ai/download](https://ollama.ai/download).

Then pull the Mistral model:

```bash
ollama pull mistral
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🧮 How It Works

1. **Scraper Module** fetches and cleans the documentation pages for all supported ML libraries.
2. **FAISS Index** stores vector embeddings of the docs for fast similarity search.
3. When a user asks a question:
   * The relevant context is retrieved from FAISS.
   * Ollama Mistral generates an answer using that context.
4. **Memory Module** preserves the last few interactions for better conversation continuity.

---

## 🧠 Example Queries

* "How do I normalize data using scikit-learn?"
* "Show me how to plot a bar chart in Seaborn."
* "What is the difference between `fit_transform()` and `transform()` in sklearn?"
* "How can I fetch live stock prices using yFinance?"

---

## 🖼️ Screenshots

<!-- Add screenshots or screen recordings of your Streamlit interface here -->
![App Screenshot](path/to/screenshot.png)

---

## 🧰 Future Improvements

- [ ] Add user authentication
- [ ] Support for more ML frameworks (PyTorch, XGBoost, etc.)
- [ ] Export chat history
- [ ] Integration with cloud storage for larger FAISS indexes
- [ ] Multi-language support
- [ ] Custom model fine-tuning options

---

## 🧑‍💻 Author

**Vaibhav Barala**

🔗 [LinkedIn](https://linkedin.com/in/vaibhavbarala)  
📧 vaibhavbarala8@gmail.com  
🐙 [GitHub](https://github.com/vaibhavbarala26)

---

## 🪪 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* [Ollama](https://ollama.ai/) for providing local LLM inference
* [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
* [Streamlit](https://streamlit.io/) for the amazing web framework
* All the open-source ML libraries whose documentation makes this project possible

---

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


<div align="center">
Made with ❤️ by Vaibhav Barala
</div>
```

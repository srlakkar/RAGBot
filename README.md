# 🌸 Flower Image RAG Demo

A **Streamlit-based demo** that combines **image embeddings**, **caption generation (BLIP-2)**, **vector search (Qdrant)**, and **LLM reasoning (OpenAI GPT-4o)** to provide context-aware answers about flower images.

This project demonstrates an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** for visual queries with textual reasoning.

---

## 🚀 Features

* **Automatic Captioning:** Upload a flower image and generate a descriptive caption using **BLIP-2**, highlighting type, color, and surroundings.
* **Visual Similarity Search:** Find visually similar flowers via **ResNet50 embeddings** and **Qdrant vector database**.
* **Neighbor Insights:** Display similar images along with captions to provide contextual information.
* **Question Answering:** Ask natural language questions about the uploaded flower and receive detailed, botanist-friendly answers powered by **GPT-4o**.
* **Retrieval-Augmented Generation:** Combines visual context from neighbors with LLM reasoning for **rich, informed answers**.

---

## 🛠 Architecture Overview

1. **Image Upload:** Users upload a flower image via Streamlit UI.
2. **Caption Generation:** BLIP-2 generates a detailed caption for the query image.
3. **Vector Embeddings:** ResNet50 converts images into embeddings for similarity search.
4. **Neighbor Retrieval:** Qdrant finds the top visually similar images, excluding the query image itself.
5. **Context-Aware LLM Response:** Captions of neighbor images are combined with the query caption and fed into GPT-4o to answer user questions.
6. **Display:** Streamlit shows:

   * Query image and caption
   * Neighbor images with captions
   * GPT-4o’s answer to the user question

---

## 📁 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/flower-image-rag.git
cd flower-image-rag
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your **OpenAI API key** in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

5. Run the app:

```bash
streamlit run app.py
```

---

## 🎨 Usage

1. Upload a flower image (`.jpg`, `.jpeg`, `.png`).
2. Wait for BLIP-2 to generate a caption.
3. Explore visually similar flowers retrieved from Qdrant.
4. Type a natural language question about your flower and receive a GPT-4o-generated answer.

---

## ⚡ Notes

* **GPU Recommended:** BLIP-2 runs faster on GPU, but CPU is supported.
* **Vector DB:** Qdrant is used for similarity search; ensure it’s running locally (`localhost:6333`).
* **Extensible:** You can swap BLIP-2 for other captioning models or GPT-4o for other LLMs.
* **Caching:** Streamlit caching is used to avoid recomputing embeddings and captions.

---

## 📈 Potential Improvements

* Fine-tune BLIP-2 on a flower dataset for more precise captions.
* Add multi-language support for captions and LLM responses.
* Expand RAG pipeline to include **species identification** using domain-specific LLM prompts.
* Integrate **larger image datasets** in Qdrant for more diverse neighbor retrieval.

---

## 🖼 Screenshots

> *(Include sample images of the app with query image, neighbors, and LLM response here.)*

---

## 🔗 References

* [BLIP-2: Bootstrapping Language-Image Pre-training](https://huggingface.co/models)
* [Qdrant Vector Database](https://qdrant.tech/)
* [OpenAI GPT-4o](https://platform.openai.com/)
* [Streamlit Documentation](https://docs.streamlit.io/)



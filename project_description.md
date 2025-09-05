# Project Description: Flower Image RAG Demo

This project demonstrates how **Retrieval-Augmented Generation (RAG)** can be applied to **image-based tasks**. The goal is to enable users to upload an image of a flower, retrieve visually similar images, and generate **context-aware natural language answers** using a Large Language Model.

---

## 1. Objectives
- Show how multimodal pipelines (vision + language) can be combined in a simple app.
- Use **image captioning (BLIP-2)** to bridge the gap between vision and text. 
- Store image embeddings and captions in a **vector database (Qdrant)** for similarity search.
- Efficiently searches similar images and identifies **top-n nearest neighbors**.
- Use **LLMs (OpenAI GPT-4o)** to create detailed natural language responses informed by retrieved contexts and user queries.

This pipeline is useful for botanists, gardeners, and hobbyists seeking accurate flower identification and advice from visual data.

---

## 2. Architecture

### Components
1. **ResNet50**  
   - Pretrained CNN used to extract 2048-dimensional embeddings of input images.
   - Embeddings are stored in and queried from Qdrant.

2. **Qdrant Vector Database**  
   - Stores embeddings + metadata (file path, caption).
   - Supports fast nearest neighbor search.

3. **BLIP-2 (Bootstrapping Language-Image Pretraining)**  
   - Generates natural language captions for images.
   - Used for both query images and neighbors.

4. **OpenAI GPT-4o**  
   - Takes captions as context.
   - Answers user questions in natural language.

5. **Streamlit UI**  
   - Simple web interface for uploading images, displaying neighbors, and interacting with GPT.

---

## 3. Workflow
1. **Image Upload**  
   - User uploads a flower image in the Streamlit app.  
   - BLIP generates a caption.  
   - ResNet50 produces an embedding.

2. **Neighbor Retrieval**  
   - Embedding is used to query Qdrant.  
   - Top `N` similar images are returned.  
   - If captions are missing, BLIP generates them.

3. **Question Answering**  
   - User enters a natural language question.  
   - A prompt is built from the query caption + neighbor captions.  
   - Prompt is passed to GPT-4o.  
   - The model answers as a "helpful botanist."

---

## 4. Example Use Case
- **Upload Image**: a rose photo  
- **Generated Caption**: "a red rose in bloom with green leaves"  
- **Retrieved Neighbors**: similar rose photos with captions  
- **User Question**: "How do I care for this flower?"  
- **LLM Answer**: GPT explains watering frequency, sunlight needs, and common rose care tips.

---

## 5. Future Improvements
- Add **image upload dataset indexing tool** (to populate Qdrant easily).  
- Add support for **BLIP-2** or **CLIP** for stronger captioning/embedding.  
- Multi-image query support (compare flowers).  
- Expand beyond flowers → wildlife, products, medical images, etc.  
- Adjustable **LLM response length** (via Streamlit slider).  

---

## 6. Conclusion
This project is a working demonstration of multimodal RAG, showing how **vision (images)** and **language (captions + LLM)** can be combined to create interactive, AI-powered applications. It serves as both a learning resource and a base for extending into domain-specific solutions.

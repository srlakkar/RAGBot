# Project Description: GardenGPT

This project demonstrates how **Retrieval-Augmented Generation (RAG)** can be applied to **image-based tasks**. The goal is to enable users to upload an image of a flower, retrieve visually similar images, and generate **context-aware natural language answers** using a Large Language Model (LLM).

---

## 1. Objectives

* Demonstrate how **multimodal AI pipelines** (vision + language) can be integrated into a simple, interactive web app.
* Use **image captioning (BLIP-2)** to bridge the gap between visual content and textual representation.
* Store image embeddings and captions in a **vector database (Qdrant)** for fast and efficient similarity search.
* Retrieve **top-n nearest neighbors** to provide contextual visual references.
* Leverage **LLMs (OpenAI GPT-4o)** to generate detailed natural language responses informed by retrieved contexts and user queries.

This pipeline is useful for **botanists, gardeners, hobbyists**, or anyone seeking AI-assisted visual analysis and advice from flower images.

---

## 2. Prerequisites

* Python 3.10 or higher
* Optional GPU with CUDA support for faster inference
* Qdrant vector database installed and running locally
* OpenAI API key
* Dependencies installed via `requirements.txt`

---

## 3. Architecture

### Components

1. **ResNet50**

   * Pretrained CNN for extracting 2048-dimensional embeddings of images.
   * Embeddings serve as the vector representation for similarity search.

2. **Qdrant Vector Database**

   * Stores embeddings and metadata (file path, caption).
   * Performs fast nearest neighbor searches for retrieval of visually similar images.

3. **BLIP-2 (Bootstrapped Language-Image Pretraining)**

   * Generates natural language captions describing the content of images.
   * Used for both the query image and any retrieved neighbor images.

4. **OpenAI GPT-4o**

   * Takes captions as context along with user questions.
   * Produces detailed, domain-specific responses as a "helpful botanist."

5. **Streamlit UI**

   * Provides a user-friendly web interface for image upload, neighbor display, and natural language interaction with GPT.

### Workflow Diagram (ASCII)

```
User Upload --> BLIP-2 Caption --> ResNet50 Embedding --> Qdrant Search --> Neighbors Captions --> GPT-4o --> Response
```

---

## 4. Workflow

1. **Image Upload**

   * User uploads a flower image via Streamlit.
   * BLIP-2 generates a descriptive caption.
   * ResNet50 produces a vector embedding.

2. **Neighbor Retrieval**

   * Embedding is used to query Qdrant.
   * Top `N` visually similar images are retrieved.
   * Captions for neighbors are generated on-the-fly if missing.

3. **Question Answering**

   * User submits a natural language question.
   * A prompt is constructed using the query image caption and neighbor captions.
   * GPT-4o generates a context-aware answer based on the retrieved visual information.

---

## 5. Example Use Case

* **Upload Image**: Photo of a red rose.
* **Generated Caption**: "A red rose in bloom with green leaves."
* **Retrieved Neighbors**: Similar rose images with descriptive captions.
* **User Question**: "How do I care for this flower?"
* **LLM Answer**: GPT provides guidance on watering, sunlight, soil, and pruning tips specific to roses.

---

## 6. Future Improvements

* Add a **dataset indexing tool** for bulk image uploads to populate Qdrant automatically.
* Support alternative embeddings or captioning models (e.g., **CLIP**, fine-tuned BLIP-2) for better accuracy.
* Enable **multi-image queries** to compare multiple flowers at once.
* Extend the RAG pipeline to other domains such as wildlife, medical images, or product catalogs.
* Introduce adjustable **LLM response parameters** (length, style) through the UI.
* Improve caption quality with domain-specific fine-tuning of BLIP-2.

---

## 7. Limitations

* Caption quality may vary; BLIP-2 can sometimes produce generic descriptions.
* Large models require GPU or are slow on CPU.
* Small vector dataset may limit neighbor retrieval accuracy.
* Currently only supports flower images.

---

## 8. Data Privacy and Ethics

* No personal images are stored permanently.
* Users are expected to upload images of flowers only.
* All processing happens locally (except LLM queries to OpenAI).

---

## 9. Conclusion

This project demonstrates a **working multimodal RAG system**, combining **vision (images)** and **language (captions + LLM)** to create an interactive AI-powered application. It serves as both a **learning resource** and a **foundation for domain-specific extensions**, offering practical insights into how RAG can enhance visual understanding and contextual reasoning.

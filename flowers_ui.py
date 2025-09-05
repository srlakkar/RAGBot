import streamlit as st
from PIL import Image
import tempfile
import os

from pipeline import (
    get_embedding,
    get_top_neighbors,
    generate_caption_blip,
    query_gpt4,
    client,
    collection_name,
)

st.title("🌸 Flower Image RAG Demo")
st.write("Upload a flower image, see similar flowers, and ask context-aware questions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
user_question = st.text_input("Ask a question about the flower:")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        query_image_path = tmp.name

    # Show uploaded image
    st.image(query_image_path, caption="Query Image", width='stretch')

    with st.spinner("Processing..."):
        embedding = get_embedding(query_image_path)
        neighbors = get_top_neighbors(embedding, client, collection_name, top_n=5)
        query_caption = generate_caption_blip(query_image_path)

    # --- Show query caption ---
    st.subheader("📝 Query Image Caption")
    st.info(query_caption or "No caption generated.")

    # --- Show neighbors and captions ---
    st.subheader("🔍 Similar Images")
    cols = st.columns(len(neighbors))
    for i, (col, neighbor) in enumerate(zip(cols, neighbors)):
        if os.path.exists(neighbor["filepath"]):
            col.image(neighbor["filepath"], caption=f"Neighbor {i+1}")
        # Show caption (generate if missing)
        if neighbor["caption"]:
            col.caption(neighbor["caption"])
        else:
            try:
                cap = generate_caption_blip(neighbor["filepath"])
                col.caption(cap)
            except Exception:
                col.caption("No caption available.")

    # --- If user asked a question, query LLM ---
    if user_question:
        with st.spinner("Querying GPT-4..."):
            context_lines = [f"- {item['caption']}" for item in neighbors if item["caption"]]
            context_lines.insert(0, f"[Query Image Caption] {query_caption}")
            context_text = "\n".join(context_lines)

            prompt = (
                f"Here are the captions of flower images visually similar to the query image:\n"
                f"{context_text}\n\n"
                f"User question: {user_question}\n"
                f"Answer as a helpful botanist."
            )
            llm_response = query_gpt4(prompt, max_tokens=800)

        st.subheader("🤖 GPT-4 Response")
        st.success(llm_response)

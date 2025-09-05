import streamlit as st
from PIL import Image
import tempfile, os

from pipeline import (
    get_embedding,
    get_top_neighbors,
    generate_caption_blip,
    query_gpt4,
    client,
    collection_name,
)

# --- Page setup: wide layout ---
st.set_page_config(layout="wide")

# --- Centered title and subtitle ---
st.markdown("<h1 style='text-align: center;'>🌸 FGardenGPT RAG Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Upload a flower image, see similar flowers, and ask context-aware questions.</p>",
    unsafe_allow_html=True
)

# --- Two-column layout ---
left_col, right_col = st.columns([1,2])

with left_col:
    st.subheader("📝 Choose an image...")
    uploaded_file = st.file_uploader("",type=["jpg", "jpeg", "png"])
    user_question = st.text_input("Ask a question about the flower:")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            query_image_path = tmp.name

        # Show query image
        st.image(query_image_path, caption="Query Image", use_container_width=True)

with right_col:
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            embedding = get_embedding(query_image_path)
            neighbors = get_top_neighbors(embedding, client, collection_name, top_n=5)
            query_caption = generate_caption_blip(query_image_path)

        # --- Query image caption ---
        st.subheader("📝 Query Image Caption")
        st.info(query_caption or "No caption generated.")

        # --- Show neighbors in rows of 2 or 3 depending on number ---
        st.subheader("🔍 Similar Images")
        if neighbors:
            num_per_row = 5  # adjust: 2 or 3 images per row
            for i in range(0, len(neighbors), num_per_row):
                row = st.columns(num_per_row, gap="large")
                for col, neighbor in zip(row, neighbors[i:i+num_per_row]):
                    if os.path.exists(neighbor["filepath"]):
                        col.image(neighbor["filepath"], caption="Similar flower", width='stretch')
                    if neighbor["caption"]:
                        col.caption(neighbor["caption"])
                    else:
                        try:
                            cap = generate_caption_blip(neighbor["filepath"])
                            col.caption(cap)
                        except Exception:
                            col.caption("No caption available.")

        # --- Query GPT-4 if user entered a question ---
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



#streamlit: builds the web UI.
#tempfile + os: used to save and access the uploaded file on disk.
#pipeline imports are user-defined helper functions / objects that provide the heavy lifting:
#get_embedding(image_path) — returns a numeric embedding (vector) for the image.
#get_top_neighbors(embedding, client, collection_name, top_n=5) — queries a vector DB (using client and collection_name) and returns the closest images/metadata.
#generate_caption_blip(image_path) — produces a natural-language caption for an image (BLIP model or similar).
#query_gpt4(prompt, max_tokens=...) — sends the prompt to an LLM (GPT-4) and returns text.
#client, collection_name — configuration/connection for your vector DB.

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
st.markdown("<h1 style='text-align: center; margin-bottom:0;'>🌸 GardenGPT RAG Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:18px; color:gray;'>Upload a flower image, explore similar flowers, and ask context-aware questions.</p>",
    unsafe_allow_html=True
)

st.write("")  # spacer

# --- Two-column layout ---
left_col, right_col = st.columns([1.2, 2], gap="large")

with left_col:
    # --- Upload Section ---
    with st.container():
        st.markdown("### 📂 Choose an Image")
        uploaded_file = st.file_uploader("Upload flower image", type=["jpg", "jpeg", "png"])

        st.markdown("### 💬 Ask a Question")
        user_question = st.text_input("Type your question here...")

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                query_image_path = tmp.name

            # Show query image
            st.image(query_image_path, caption="Query Image", use_container_width=True)

with right_col:
    if uploaded_file is not None:
        with st.spinner("🔎 Analyzing image..."):
            embedding = get_embedding(query_image_path)
            neighbors = get_top_neighbors(embedding, client, collection_name, top_n=5)
            query_caption = generate_caption_blip(query_image_path)

        # --- Query Image Caption (Card style) ---
        with st.container():
            st.markdown("### 📝 Query Image Caption")
            st.info(query_caption or "No caption generated.")

        # --- Similar Images (Grid style) ---
        with st.container():
            st.markdown("### 🔍 Similar Images")
            if neighbors:
                num_per_row = 5
                for i in range(0, len(neighbors), num_per_row):
                    row = st.columns(num_per_row, gap="medium")
                    for col, neighbor in zip(row, neighbors[i:i+num_per_row]):
                        if os.path.exists(neighbor["filepath"]):
                            col.image(neighbor["filepath"], caption="", use_container_width=True)
                        if neighbor["caption"]:
                            col.caption(f"_{neighbor['caption']}_")
                        else:
                            try:
                                cap = generate_caption_blip(neighbor["filepath"])
                                col.caption(f"_{cap}_")
                            except Exception:
                                col.caption("_No caption available_")

        # --- GPT-4 Response (Chat style) ---
        if user_question:
            with st.spinner("🤖 Thinking..."):
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

            with st.container():
                st.markdown("### 🤖 GPT-4 Response")
                with st.chat_message("user"):
                    st.write(user_question)
                with st.chat_message("assistant"):
                    st.write(llm_response)

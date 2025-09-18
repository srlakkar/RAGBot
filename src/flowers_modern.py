import streamlit as st
from PIL import Image
import tempfile, os
import hashlib

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
st.markdown("<h1 style='text-align: center;'>🌸 GardenGPT Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Upload a flower image, see similar flowers, and ask context-aware questions.</p>",
    unsafe_allow_html=True
)

# --- Helpers to manage session state ---
def clear_session():
    for k in ["uploaded_file_hash", "query_image_path", "neighbors", "query_caption", "llm_response"]:
        if k in st.session_state:
            del st.session_state[k]

# --- Layout: left controls, right results ---
left_col, right_col = st.columns([1.2, 2], gap="medium")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📂 Choose an Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">💬 Ask a Question</div>', unsafe_allow_html=True)
    user_question = st.text_input("", placeholder="e.g. What species is this?", key="question")

    # Clear button
    if st.button("Clear"):
        clear_session()
        # reset file_uploader widget by rerunning (so uploader visually clears)
        st.experimental_rerun()

    # Handle upload and compute neighbors only when a new file is uploaded (detect via hash)
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        needs_processing = st.session_state.get("uploaded_file_hash") != file_hash

        if needs_processing:
            # Save file to a temp path and persist it in session_state
            suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(file_bytes)
            tmp.close()
            st.session_state["query_image_path"] = tmp.name
            st.session_state["uploaded_file_hash"] = file_hash

            # compute embedding, neighbors, and captions ONCE
            with st.spinner("🔎 Analyzing image ..."):
                embedding = get_embedding(st.session_state["query_image_path"])
                neighbors = get_top_neighbors(embedding, client, collection_name, top_n=5) or []
                # Ensure each neighbor has a caption (generate if missing) and keep filepath absolute
                for n in neighbors:
                    # If path is relative or missing, leave as-is; assume existing pipeline returns valid filepath
                    if not n.get("caption"):
                        try:
                            if n.get("filepath") and os.path.exists(n["filepath"]):
                                n["caption"] = generate_caption_blip(n["filepath"])
                            else:
                                n["caption"] = "No caption available"
                        except Exception:
                            n["caption"] = "No caption available"
                # persist neighbors and query_caption
                st.session_state["neighbors"] = neighbors
                try:
                    st.session_state["query_caption"] = generate_caption_blip(st.session_state["query_image_path"])
                except Exception:
                    st.session_state["query_caption"] = "No caption generated."
                # clear any previous llm response so user sees new context if they ask again
                if "llm_response" in st.session_state:
                    del st.session_state["llm_response"]

        # show preview (always use stored path so it doesn't force reprocessing)
        if "query_image_path" in st.session_state and os.path.exists(st.session_state["query_image_path"]):
            st.image(st.session_state["query_image_path"], caption="Query Image", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # If nothing uploaded yet, show placeholder
    if "query_image_path" not in st.session_state:
        st.markdown('<div class="card" style="min-height:220px;display:flex;align-items:center;justify-content:center;">'
                    '<div style="text-align:center;"><div style="font-weight:600;margin-bottom:6px">Ready when you are</div>'
                    '<div class="muted">Upload an image on the left to compute neighbors and captions.</div></div></div>',
                    unsafe_allow_html=True)
    else:
        # --- Query caption (comes from session_state) ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📝 Query Image Caption</div>', unsafe_allow_html=True)
        st.info(st.session_state.get("query_caption", "No caption generated."))
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Similar Images: show top 5 side-by-side; DO NOT recompute neighbors here ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🔍 Similar Images</div>', unsafe_allow_html=True)

        neighbors = st.session_state.get("neighbors") or []
        if neighbors:
            cols = st.columns(5, gap="small")
            for col, neighbor in zip(cols, neighbors[:5]):
                with col:
                    fp = neighbor.get("filepath")
                    if fp and os.path.exists(fp):
                        st.image(fp, use_container_width=True)
                    else:
                        # if filepath missing or invalid, show placeholder box
                        st.write("")  # keep layout
                    # caption already ensured during initial processing
                    cap = neighbor.get("caption") or "No caption available"
                    if len(cap) > 48:
                        cap = cap[:45] + "..."
                    st.markdown(f"<div class='small-caption'>_{cap}_</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="muted">No similar images found.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- GPT-4 Response: uses stored neighbors and caption; does NOT recompute neighbors ---
        if user_question:
            # If we already have a cached response for the same question, reuse it (optional)
            cached_key = f"llm_response_{hashlib.sha256(user_question.encode()).hexdigest()}"
            if cached_key in st.session_state:
                llm_response = st.session_state[cached_key]
            else:
                with st.spinner("🤖 Thinking (LLM call)..."):
                    context_lines = [f"- {n.get('caption')}" for n in neighbors if n.get("caption")]
                    context_lines.insert(0, f"[Query Image Caption] {st.session_state.get('query_caption','')}")
                    context_text = "\n".join(context_lines)

                    prompt = (
                        f"Here are the captions of flower images visually similar to the query image:\n"
                        f"{context_text}\n\n"
                        f"User question: {user_question}\n"
                        f"Answer as a helpful botanist."
                    )
                    llm_response = query_gpt4(prompt, max_tokens=800)
                    # cache per-question response during the session
                    st.session_state[cached_key] = llm_response

            # render response card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🤖 GPT-4 Response</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat">', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble user">{user_question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble assistant">{llm_response}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# subtle footer
st.markdown("<div style='padding:8px 0 16px 0; text-align:center; color: #6b7280; font-size:12px;'>© GardenGPT — UI optimized to avoid recomputing neighbors on question submission</div>", unsafe_allow_html=True)

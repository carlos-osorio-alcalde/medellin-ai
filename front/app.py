import streamlit as st
from configuration.load import config
from rag.clients import qdrant_manager
from rag import search_similar_items
from rag.core.models import create_caption, Modalities
import time

# Custom theme
st.set_page_config(
    page_title="Multimodal Animal Retrieval",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        "Get Help": "https://www.example.com/help",
        "Report a bug": "https://www.example.com/bug",
        "About": "# This is a multimodal retrieval app for animal-related content.",  # noqa
    },
)

# Custom CSS
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app demonstrates multimodal retrieval for animal-related content. You can search using text, audio, or images."  # noqa
)
st.sidebar.markdown("---")
st.sidebar.subheader("How to use")
st.sidebar.markdown(
    """
1. Select a tab for your search type
2. Enter text or upload a file
3. Click the search button
4. View results below
"""
)


def display_results(results, result_type):
    if not results:
        st.warning("No results found.")
        return

    st.subheader(f"Retrieved {result_type} results:")
    cols = st.columns(3)
    for i, result in enumerate(results):
        with cols[i % 3]:
            st.markdown(f"**Result {i+1}**")
            if result_type == "audio":
                st.audio(
                    "data/audio/" + result.payload["filename"], start_time=0
                )
            elif result_type == "image":
                st.image(
                    "data/images/" + result.payload["filename"], width=400
                )
            st.write(f"Similarity: {result.score:.4f}")
            st.write(f"Filename: {result.payload['filename']}")
            st.markdown("---")


# Title of the app
st.title("🐾 Multimodal Animal Retrieval")

# Create tabs with icons
tab1, tab2, tab3 = st.tabs(["📝 Text", "🎵 Audio", "🖼️ Image"])

with tab1:
    text_query = st.text_input("Enter a text query:")
    if st.button("🔍 Search Text", use_container_width=True):
        if text_query:
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            with st.spinner("Searching..."):
                results_audio = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_audio"],
                    value=text_query,
                )
                results_images = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_image"],
                    value=text_query,
                )
                display_results(results_audio, "audio")
                display_results(results_images, "image")
        else:
            st.warning("Please enter a text query.")

with tab2:
    audio_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if audio_file:
        st.audio(audio_file)
    if st.button("🔍 Search Audio", use_container_width=True):
        if audio_file:
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            audio_path = "temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())

            with st.spinner("Searching audio..."):
                results = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_audio"],
                    value=audio_path,
                )
                display_results(results, "audio")

            with st.spinner("Retrieving images..."):
                caption = create_caption(
                    modality=Modalities.AUDIO, path=audio_path
                )
                results_images = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_image"],
                    value=caption,
                )
                display_results(results_images, "image")
        else:
            st.warning("Please upload an audio file.")

with tab3:
    image_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if image_file:
        st.image(image_file, width=300)
    if st.button("🔍 Search Image", use_container_width=True):
        if image_file:
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_file.read())

            with st.spinner("Searching images..."):
                results = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_image"],
                    value=image_path,
                )
                display_results(results, "image")

            with st.spinner("Retrieving audios..."):
                caption = create_caption(
                    modality=Modalities.IMAGE, path=image_path
                )
                result_audio = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_audio"],
                    value=caption,
                )
                display_results(result_audio, "audio")
        else:
            st.warning("Please upload an image file.")

# Add a clear button
if st.button("🔄 Clear", use_container_width=True):
    st.experimental_rerun()

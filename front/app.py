import streamlit as st
from configuration.load import config
from rag.clients import qdrant_manager
from rag import search_similar_items
from rag.core.models import create_caption, Modalities

st.set_page_config(
    page_title="A toy multimodal RAG",
    initial_sidebar_state="auto",
    layout="wide",
)


def display_results(results, result_type):
    if not results:
        st.write("No results found.")
        return

    st.subheader(f"Retrieved results {result_type}:")
    cols = st.columns(3)
    for i, result in enumerate(results):
        with cols[i % 3]:
            st.markdown(f"**Result {i+1}**")
            if result_type == "audio":
                st.audio("data/audio/" + result.payload["filename"])
            elif result_type == "image":
                st.image(
                    "data/images/" + result.payload["filename"], width=400
                )

            st.write(f"Similarity: {result.score:.4f}")
            st.write(f"Filename: {result.payload['filename']}")
            st.markdown("---")


# Title of the app
st.title("A toy multimodal RAG: let's play with animals")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Text", "Audio", "Image"])

with tab1:
    text_query = st.text_input("Enter a text query:")
    if st.button("Search Text", use_container_width=True):
        if text_query:
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
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.audio(audio_file)
    if st.button("Search Audio", use_container_width=True):
        if audio_file:
            audio_path = "temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())

            results = search_similar_items(
                qdrant_manager=qdrant_manager,
                collection_name=config["vectordb"]["collection_audio"],
                value=audio_path,
            )

            display_results(results, "audio")

            # Get the caption of the audio
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
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.image(image_file, width=600)
    if st.button("Search Image", use_container_width=True):
        if image_file:
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_file.read())

            results = search_similar_items(
                qdrant_manager=qdrant_manager,
                collection_name=config["vectordb"]["collection_image"],
                value=image_path,
            )

            display_results(results, "image")

            # Get the caption of the audio
            with st.spinner("Retrieving audios..."):
                caption = create_caption(
                    modality=Modalities.IMAGE, path=image_path
                )
                print(caption)
                result_audio = search_similar_items(
                    qdrant_manager=qdrant_manager,
                    collection_name=config["vectordb"]["collection_audio"],
                    value=caption,
                )
                display_results(result_audio, "audio")
        else:
            st.warning("Please upload an image file.")


# Add a clear button
if st.button("Clear", use_container_width=True):
    st.experimental_rerun()

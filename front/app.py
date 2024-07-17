import streamlit as st
from rag.clients import qdrant_manager
from rag import search_similar_audios, create_text_from_audio

st.set_page_config(
    page_title="Multimodal RAG, Audio-Text",
    initial_sidebar_state="auto",
)


# Title of the app
st.title("Multimodal RAG: Audio-Text")

# Create a text input for the text query
text_query = st.text_input("Enter text query:")

# Create a file uploader for the audio file
audio_file = st.file_uploader("Upload a .wav file", type=["wav"])

# If the audio file is uploaded, display the audio
if audio_file:
    st.audio(audio_file)

# Process the input
if st.button("Search", use_container_width=True):
    if text_query:
        # Search using text query
        results = search_similar_audios(qdrant_manager, text_query)
    elif audio_file:
        # Save the uploaded audio file temporarily
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        # Text with the fun fact about the audio
        with st.spinner(
            "Searching for similar audios and generating fun fact..."
        ):
            # Search using audio file path
            results = search_similar_audios(qdrant_manager, audio_path)
            fun_fact = create_text_from_audio(audio_path)
    else:
        results = None
        st.warning("Please enter a text query or upload an audio file.")

    # Display the results
    if results:
        st.write("Retrieved Audios:")
        for result in results:
            st.audio("data/audio/" + result.payload["filename"])

        st.write(fun_fact)

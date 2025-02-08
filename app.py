import streamlit as st
from prototype import transcribe_audio, generate_subtitles, extract_audio, marianmt_translate, cleanup_files
from lang_config import LANGUAGE_MAP
import time
import os
import boto3

# Streamlit app
def app():
    st.title("Indie AI")
    st.caption("by AI Alchemy")
    st.write("⚡ Running on CPU")
    st.write("Transform your videos with AI-powered subtitles in Indian languages.")

    # Upload video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    # Show full language names in UI, but use codes internally
    lang_names = list(LANGUAGE_MAP.values())
    selected_lang = st.selectbox("Select target language", lang_names)
    # Convert back to language code
    target_lang = [code for code, name in LANGUAGE_MAP.items() if name == selected_lang][0]

    if video_file is not None:
        # Save the uploaded video
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Generate subtitles
        if st.button("Generate Subtitles"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Extracting audio...")
                base_name = os.path.splitext(video_path)[0]
                audio_path = f"{base_name}.mp3"
                extract_audio(video_path, audio_path)
                progress_bar.progress(25)
                
                status_text.text("Transcribing audio...")
                segments = transcribe_audio(audio_path)
                progress_bar.progress(50)
                
                status_text.text("Translating subtitles...")
                subtitle_path = f"{base_name}_{target_lang}.srt"
                generated_path = generate_subtitles(segments, target_lang, subtitle_path)
                subtitle_path = os.path.abspath(generated_path)  # Get absolute path
                progress_bar.progress(100)

                if subtitle_path:
                    st.success("Subtitles generated successfully!")

                    # Download subtitles
                    if os.path.exists(subtitle_path):
                        with open(subtitle_path, "r") as f:
                            content = f.read()
                            st.download_button(
                                "Download Subtitles", 
                                content, 
                                file_name=os.path.basename(subtitle_path)
                            )
                    else:
                        # Retry after short delay
                        time.sleep(0.5)
                        if os.path.exists(subtitle_path):
                            with open(subtitle_path, "r") as f:
                                st.download_button(
                                    "Download Subtitles", 
                                    f.read(), 
                                    file_name=os.path.basename(subtitle_path)
                                )
                            st.error("Failed to generate downloadable subtitles. Please try again.")

                    # Cleanup after download
                    cleanup_files(video_path, audio_path, subtitle_path)
                else:
                    st.error("Failed to generate subtitles. Please check the logs.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

                # Cleanup on error
                cleanup_files(video_path, audio_path, subtitle_path)

# Hidden backup option
def hidden_main():
    audio_path = "uploaded_video.mp4"
    segments = transcribe_audio(audio_path, use_aws=False)
    subs = marianmt_translate(segments)

if __name__ == "__main__":
    app()
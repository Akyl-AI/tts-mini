import streamlit as st
import subprocess

def run_tts():
    st.title("Matcha TTS Interface")


    speaking_rate = st.slider("Select Speaking Rate", min_value=0.1, max_value=2.0, value=0.8, step=0.1)


    text = st.text_input("Enter text for speech synthesis:")


    if st.button("Generate Speech"):
        if text:  
            command = f"matcha-tts --speaking_rate {speaking_rate} --text \"{text}\""

            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            

            audio_filename = "utterance_001.wav"
            

            if result.returncode == 0: 
                st.success("Speech generated successfully:")
                # Воспроизведение аудио
                audio_file = open(audio_filename, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav', start_time=0)
                audio_file.close()
            else:
                if result.stdout:
                    st.text(result.stdout)
                if result.stderr:
                    st.error("Error in command execution:")
                    st.text(result.stderr)
        else:
            st.error("Please enter text to synthesize.")


if __name__ == "__main__":
    run_tts()

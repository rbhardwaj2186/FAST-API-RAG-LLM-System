import streamlit as st
import requests

# Title of the Streamlit app
st.title("Upload a File to FastAPI")

# File uploader widget
file = st.file_uploader("Choose a file", type=["pdf"])

# Button to trigger file upload
if st.button("Submit"):
    if file is not None:
        # Prepare file for upload
        files = {"file": (file.name, file, file.type)}
        try:
            # Send POST request to FastAPI server
            response = requests.post("http://localhost:8001/upload", files=files)
            # Display the response from the server
            st.write(response.json())
        except Exception as e:
            st.error(f"Failed to upload file: {e}")
    else:
        st.warning("No file uploaded.")

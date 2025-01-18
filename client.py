import streamlit as st
import requests

# FastAPI server URL
API_BASE_URL = "http://localhost:8001"

# Streamlit App Title
st.title("RAG System with FastAPI and Streamlit")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Upload File", "Search", "Manage Collections"])

# Tab 1: File Upload
with tab1:
    st.header("Upload a File")
    file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if st.button("Submit File"):
        if file is not None:
            # Prepare file for upload
            files = {"file": (file.name, file, file.type)}
            try:
                # Send POST request to upload endpoint
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                st.write(response.json())
            except Exception as e:
                st.error(f"Failed to upload file: {e}")
        else:
            st.warning("Please upload a PDF file.")

# Tab 2: Search
with tab2:
    st.header("Search for Content")
    query = st.text_input("Enter your search query:")
    top_k = st.number_input("Number of results to retrieve:", min_value=1, max_value=20, value=5, step=1)
    if st.button("Search"):
        if query:
            try:
                # Prepare payload for search
                payload = {"query": query, "top_k": top_k}
                response = requests.post(f"{API_BASE_URL}/search", json=payload)
                results = response.json().get("results", [])

                if results:
                    st.write("Search Results:")
                    for i, result in enumerate(results, 1):
                        st.write(f"**Result {i}:**")
                        st.write(f"- Original Text: {result['original_text']}")
                        st.write(f"- Source: {result['source']}")
                        st.write(f"- Score: {result['score']:.2f}")
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Failed to search: {e}")
        else:
            st.warning("Please enter a search query.")

# Tab 3: Manage Collections
with tab3:
    st.header("Manage Qdrant Collections")
    action = st.selectbox("Choose an action:", ["List Collections", "Delete Collection"])
    if st.button("List Collections"):
        try:
            response = requests.get("http://localhost:8001/collections")
            if response.status_code == 200:
                # Parse and display the collections
                data = response.json()
                if "collections" in data:
                    collections = data["collections"]
                    if collections:
                        st.write("Available Collections:")
                        for collection in collections:
                            st.write(f"- {collection}")
                    else:
                        st.warning("No collections found.")
                else:
                    st.error("Unexpected response structure. 'collections' key not found.")
            else:
                st.error(f"Failed to list collections: {response.json()}")
        except Exception as e:
            st.error(f"Error fetching collections: {e}")
    elif action == "Delete Collection":
        collection_name = st.text_input("Enter the collection name to delete:")
        if st.button("Delete Collection"):
            if collection_name:
                try:
                    response = requests.delete(f"{API_BASE_URL}/delete-collection/{collection_name}")
                    st.write(response.json())
                except Exception as e:
                    st.error(f"Failed to delete collection: {e}")
            else:
                st.warning("Please enter a collection name.")

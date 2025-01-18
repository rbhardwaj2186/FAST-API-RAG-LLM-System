# FastAPI RAG-Based Semantic Search System
![1_1_TSOXAZp7y32J19hkyyOw](https://github.com/user-attachments/assets/1ad9e65c-dfb8-4f1c-85b1-74e8dccac9f6)






https://github.com/user-attachments/assets/73eaf19c-0945-49ca-b5de-fa63d5c86435




https://github.com/user-attachments/assets/81d40f9e-e347-4778-a77a-16832a45d5ae

bfe16664718c


## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using **FastAPI**, **Qdrant**, and **DistilBERT** for semantic search and retrieval of relevant document snippets. It is designed to handle large-scale document embeddings and enable efficient search functionalities with business impact.

---

## Features
1. **Upload Documents**: Users can upload PDF documents for processing and storage.
2. **Semantic Search**: Provides relevant document snippets for a given query using vector similarity search.
3. **Dynamic Collection Management**: Create, delete, and manage vector collections.
4. **Streamlit Client**: A user-friendly web interface for uploading documents and performing searches.

---

## Architecture
1. **FastAPI**: Backend server managing endpoints for upload, search, and collection operations.
2. **Qdrant**: Vector database for managing and searching document embeddings.
3. **DistilBERT**: Pre-trained embedding model for converting text into 768-dimensional vectors.
4. **Streamlit**: Frontend for interacting with the system.

---

## Technical Details
### **Core Components**
1. **Vector Repository**
   - Handles interaction with the Qdrant database for storing and retrieving embeddings.

2. **Embedding Model**
   - Uses **DistilBERT** for generating embeddings through mean pooling.

3. **Endpoints**
   - `/upload`: Handles document uploads and processing.
   - `/search`: Performs vector similarity search.
   - `/delete-collection`: Deletes vector collections.

4. **Streamlit Interface**
   - Uploads documents to the FastAPI backend.
   - Initiates semantic search queries.

---

## Deployment
The application can be deployed using Docker and AWS or GCP for scalability.

1. **Dockerization**
   - Use a `Dockerfile` to package the application.

2. **AWS Deployment**
   - Push the Docker image to Amazon Elastic Container Registry (ECR).
   - Deploy using AWS ECS or Elastic Beanstalk.

3. **GCP Deployment**
   - Use Google Cloud Run for serverless containerized deployment.

---

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo.git](https://github.com/rbhardwaj2186/FAST-API-RAG-LLM-System.git )
   cd your-repo

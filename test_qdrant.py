import requests

def test_qdrant_connection():
    try:
        response = requests.get("http://localhost:6333/")
        print("Connected to Qdrant successfully!")
        print("Response:", response.json())
    except Exception as e:
        print("Failed to connect to Qdrant:", e)

if __name__ == "__main__":
    test_qdrant_connection()

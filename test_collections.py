import requests

def list_collections():
    url = "http://127.0.0.1:6333/collections"
    response = requests.get(url)
    if response.status_code == 200:
        collections = response.json()["result"]["collections"]
        if collections:
            print("Collections available:")
            for collection in collections:
                print(f"- {collection['name']}")
        else:
            print("No collections found.")
    else:
        print("Failed to retrieve collections:", response.json())

if __name__ == "__main__":
    list_collections()

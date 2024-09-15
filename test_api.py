"""Implementation test for API"""
import requests

def test_api():
    """ Implementing API with timeout """
    try:
        response = requests.get('http://0.0.0.0:8000/', timeout=10)  # Set timeout in seconds
        assert response.status_code == 200
    except requests.Timeout:
        print("The request timed out")
    except requests.ConnectionError:
        print("Failed to connect to the server")
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_api()

import requests  # For sending HTTP requests
import json # For handling JSON data

# ==================== Configuration ====================
# Replace with your actual Mistral API key
API_KEY = "insert your key"

# Base URL for Mistral's OpenAI-compatible chat completions endpoint
API_URL = "https://api.mistral.ai/v1/chat/completions"

# Choose one of the available models (e.g., 'mistral-tiny', 'mistral-small', 'mistral-medium')
MODEL = "mistral-small"

# Headers for authentication and content type
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ==================== Function to Send a Chat Message ====================
def chat_with_mistral(messages, temperature):
    """
    Sends a list of messages to the Mistral chat API and returns the assistant's response.

    Parameters:
        messages (list): A list of message dictionaries in the OpenAI chat format.
                         Example: [{"role": "user", "content": "Hello!"}]

    Returns:
        str: The assistant's reply as a string.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,     # Creativity level (0 = deterministic, 1 = more random)
        "top_p": 1.0,           # Nucleus sampling parameter
        "stream": False         # Disable streaming for simple usage
    }

    # Send a POST request to Mistral's API
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))

    # Raise an error if the request failed
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    # Parse the JSON response
    response_data = response.json()

    # Extract and return the assistant's reply
    return response_data['choices'][0]['message']['content']


# ==================== Example Chat ====================
if __name__ == "__main__":
    # Initialize a conversation with a greeting
    chat_history = [
        {"role": "user", "content": "Hello, who won the last FIFA World Cup?"}
    ]

    try:
        # Send the message and receive the model's response
        response = chat_with_mistral(chat_history, 0.5)
        print("Assistant:", response)
    except Exception as e:
        print("Error:", str(e))
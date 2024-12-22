import json
import urllib3
import argparse
from tinytroupe import utils

class OllamaAPIClient:
    def __init__(self):
        self.default = {}

        self.get_config()


    def get_config(self):

        config = utils.read_config_file()

        default = {}
        default['URL'] = config["Ollama"].get('URL', 'localhost:11434')
        default['MODEL'] = config["Ollama"].get('MODEL', 'tinyllama')
        
        self.default = default

    def send_request(self, request_data, endpoint_url):
        """
        Sends a POST request with the given data to the specified URL and streams the response.

        Args:
            request_data (dict): The data to be sent as the request body.
            endpoint_url (str): The endpoint URL.

        Returns:
            list: A list of decoded JSON objects from the response.
        """
        http_manager = urllib3.PoolManager()
        request_body = json.dumps(request_data)

        response = http_manager.request(
            "POST", endpoint_url, body=request_body, preload_content=False
        )
        response_content_bytes = b""

        for response_chunk in response.stream(amt=1024):
            response_content_bytes += response_chunk
        response.release_conn()

        response_content_str = response_content_bytes.decode("utf-8").strip()
        response_json_objects = response_content_str.split("\n")
        return [json.loads(response_object) for response_object in response_json_objects]

    def extract_answer_content(self, response_json_list):
        """
        Extracts and concatenates the answer content from the JSON response.

        Args:
            response_json_list (list): A list of JSON objects containing the response messages.

        Returns:
            str: The concatenated answer content.
        """
        answer_content_list = [response_item["message"]["content"] for response_item in response_json_list]
        return "".join(answer_content_list)

def main():
    """Main function to send a question to the API and print the answer."""
    parser = argparse.ArgumentParser(description="Send a question to the API and get an answer.")
    parser.add_argument("--question", type=str, default="why is the sky blue?", required=False, help="The question to ask the API.")
    parser.add_argument("--model", type=str, default="tinyllama", help="The model to use for the API request.")
    args = parser.parse_args()

    request_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.question}],
    }

    ollama = OllamaAPIClient()
    api_url = f"{ollama.default['URL']}/api/chat"

    api_response_json = ollama.send_request(request_payload, api_url)
    extracted_answer = ollama.extract_answer_content(api_response_json)

    print("Question: ", args.question)
    print("Answer: ", extracted_answer)

if __name__ == "__main__":
    main()

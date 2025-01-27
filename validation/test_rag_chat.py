import requests
import json
from typing import Optional

class RAGConsoleClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        # Default model settings
        self.model_server = "ollama"
        self.model_name = "qwen2.5:7b-instruct-q4_0"

    def send_query(self, question: str) -> Optional[str]:
        """Send a query to the RAG API and return the response"""
        try:
            response = requests.post(
                f"{self.base_url}/recommend/",
                json={
                    "query": {"question": question},
                    "model_choice": {
                        "model_server": self.model_server,
                        "model_name": self.model_name
                    }
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None

    def run_console(self):
        """Run an interactive console session"""
        print("Wine Recommendation RAG Console")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'switch' to change model settings")
        print("-" * 50)

        while True:
            # Get user input
            user_input = input("\nYour question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            # Check for model switch command
            if user_input.lower() == 'switch':
                self.model_server = input("Enter model server (gigachat/ollama): ").strip()
                self.model_name = input("Enter model name: ").strip()
                print(f"Switched to {self.model_server} - {self.model_name}")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Send query and get response
            result = self.send_query(user_input)
            
            if result:
                print("\nResponse:", result["response"])
                print("\nRelevant contexts:")
                for context in result.get("contexts", []):
                    print("-" * 50)
                    print(context)
            else:
                print("Failed to get response from the server")

if __name__ == "__main__":
    # Create and run the console client
    client = RAGConsoleClient()
    client.run_console()

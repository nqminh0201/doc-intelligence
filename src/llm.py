"""
llm.py — Local LLM integration using Ollama.

WHAT THIS FILE DOES:
    Provides an interface to call local Large Language Models (like Llama 3.2 or Mistral)
    via the Ollama service running on your machine.

KEY CONCEPTS:
    - Local LLM: Running models directly on your hardware (CPU or GPU) without
      sending data to third-party APIs (OpenAI, Google) and with zero cost.
    - System Prompt: Instructions that define how the LLM should behave
      (e.g., "Answer truthfully based only on the provided context").
    - Temperature: Controls randomness. Low temperature (0.1 - 0.3) makes answers
      deterministic and factual (perfect for RAG). High temperature (0.7 - 1.0)
      makes answers creative and diverse.
    - Streaming: Returning text token-by-token as it is generated, instead of
      making the user wait for the entire paragraph to finish.
"""

from typing import Generator, List, Dict, Any, Optional
import ollama

# Configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL


class LLMClient:
    """
    Wrapper around the Ollama API for local LLM inference.
    """

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_BASE_URL):
        """
        Initialize the LLM client.
        
        Parameters:
            model — Name of the Ollama model (e.g. 'llama3.2', 'mistral')
            host  — Base URL where Ollama is listening (default: http://localhost:11434)
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=self.host)

    def is_available(self) -> bool:
        """
        Check if the local Ollama daemon is running and reachable.
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        """
        Return the list of models currently downloaded and available in Ollama.
        """
        try:
            response = self.client.list()
            # Handle both object and dict response structures
            models = response.get("models", []) if isinstance(response, dict) else response.models
            return [m.get("name") if isinstance(m, dict) else m.model for m in models]
        except Exception as e:
            print(f"[WARN] Failed to list models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        stream: bool = False
    ) -> Any:
        """
        Generate a completion for a prompt.

        Parameters:
            prompt        — The user query or RAG context prompt
            system_prompt — Optional system instructions
            temperature   — Creativity parameter (0.0 to 1.0). 0.2 is ideal for RAG
            stream        — If True, yields chunks of text as they arrive
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if stream:
            return self._stream_chat(messages, temperature)
        else:
            return self._sync_chat(messages, temperature)

    def _sync_chat(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """
        Synchronous chat completion: waits until the entire answer is ready.
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[ERROR] LLM generation failed: {e}"

    def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float
    ) -> Generator[str, None, None]:
        """
        Streaming chat completion: yields words/tokens as they are generated.
        """
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
                stream=True
            )
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            yield f"[ERROR] Streaming failed: {e}"


# ── Demo / Self-Check ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LOCAL LLM (OLLAMA) CLIENT CHECK")
    print("=" * 60)

    client = LLMClient()
    available = client.is_available()

    print(f"Ollama host: {client.host}")
    print(f"Target model: {client.model}")
    print(f"Ollama server running: {available}")

    if available:
        installed = client.list_installed_models()
        print(f"Installed models: {installed}")
        if client.model in installed or any(client.model in m for m in installed):
            print(f"\nTesting prompt with '{client.model}'...")
            reply = client.generate(
                prompt="Explain what a vector database is in 2 sentences.",
                temperature=0.2
            )
            print(f"\nResponse:\n{reply}")
        else:
            print(f"\n[INFO] Model '{client.model}' is not pulled yet.")
            print(f"Run: ollama pull {client.model}")
    else:
        print("\n[NOTE] Ollama server is not running on localhost:11434.")
        print("To start Ollama after installing: launch the Ollama app or run 'ollama serve'.")


if __name__ == "__main__":
    main()

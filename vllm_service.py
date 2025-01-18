import requests
import logging

logger = logging.getLogger(__name__)

# vLLM configuration
VLLM_HOST = "172.25.163.6"  # Your WSL IP
VLLM_PORT = "8000"
VLLM_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/generate"


async def generate_text(prompt: str, temperature: float = 0.7) -> str:
    """
    Generate text using vLLM service.

    Args:
        prompt (str): The input prompt
        temperature (float): Sampling temperature

    Returns:
        str: Generated text
    """
    try:
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 100  # Adjust as needed
        }

        response = requests.post(VLLM_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        return result['text']

    except Exception as e:
        logger.error(f"Error calling vLLM service: {e}")
        raise
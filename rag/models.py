import aiohttp
from loguru import logger

# System prompt to guide the model's behavior
system_prompt = "You are an AI assistant providing helpful and factual responses."

async def generate_text(prompt: str, temperature: float = 0.7) -> str:
    """
    Sends a request to the VLLM server to generate text based on the given prompt.

    Args:
        prompt (str): The input prompt for the model.
        temperature (float): The sampling temperature for text generation.

    Returns:
        str: The generated text.
    """
    # Prepare the request payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    data = {"temperature": temperature, "messages": messages}

    try:
        async with aiohttp.ClientSession() as session:
            # Send the request to the VLLM server
            async with session.post("http://localhost:8000/v1/chat", json=data) as response:
                # Check for successful response
                if response.status != 200:
                    error_message = await response.text()
                    logger.error(f"Failed to generate text: {response.status} - {error_message}")
                    return "Failed to generate text from VLLM - Check server logs for details."

                # Parse the response
                result = await response.json()
                output = result["choices"][0]["message"]["content"]
                logger.debug(f"Generated text: {output}")
                return output

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error occurred: {e}")
        return "Failed to connect to VLLM server - Check server logs for details."

    except KeyError as e:
        logger.error(f"Unexpected response format: {e}")
        return "Unexpected response format from VLLM - Check server logs for details."

    except Exception as e:
        logger.error(f"An unknown error occurred: {e}")
        return "An unknown error occurred while generating text - See server logs for details."

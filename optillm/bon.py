import logging

logger = logging.getLogger(__name__)

def best_of_n_sampling(system_prompt: str, initial_query: str, client, model: str, n: int = 3) -> str:
    print(f"best_of_n_sampling: {n}")
    token_counts = {'prompt_tokens': 0, 'completion_tokens': 0}

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}]
    
    completions = []
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        n=n,
        temperature=1
    )
    completions = [choice.message.content for choice in response.choices]
    logger.info(f"Generated {len(completions)} initial completions. Tokens used: {response.usage.completion_tokens}")
    token_counts['prompt_tokens'] += response.usage.prompt_tokens
    token_counts['completion_tokens'] += response.usage.completion_tokens
    
    # Rate the completions
    rating_messages = messages.copy()
    rating_prompt = """Rate this response from 0-10 based on reasoning quality. Give high scores (8-10) if it shows clear step-by-step logical reasoning with well-supported conclusions. Give medium scores (4-7) for partial reasoning with gaps. Give low scores (0-3) for responses without clear logical steps. Return only the numerical score."""

    rating_messages.append({"role": "system", "content": rating_prompt})
    
    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "user", "content": "Rate the above response:"})
        
        rating_response = client.chat.completions.create(
            model=model,
            messages=rating_messages,
            max_tokens=256,
            n=1,
            temperature=0.1
        )
        token_counts['prompt_tokens'] += rating_response.usage.prompt_tokens
        token_counts['completion_tokens'] += rating_response.usage.completion_tokens
        try:
            rating = float(rating_response.choices[0].message.content.strip())
            ratings.append(rating)
        except ValueError:
            ratings.append(0)
        
        rating_messages = rating_messages[:-2]
    
    best_index = ratings.index(max(ratings))
    return completions[best_index], token_counts

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import random
# import gc # Will be used in Colab

# --- TokenBanningLogitsProcessor Definition (Simplified) ---
class TokenBanningLogitsProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids: list[int]):
        super().__init__()
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id in self.banned_token_ids:
            if 0 <= token_id < scores.shape[-1]:
                scores[:, token_id] = -float('inf')
        return scores

# --- Generate Response Function ---
def generate_response(model, tokenizer, prompt_text: str, specific_banned_words: list[str] = None, random_ban_percentage: float = 0.0, max_new_tokens: int = 50):
    if specific_banned_words is None:
        specific_banned_words = []

    print(f"--- Generating Response for model: {model.config.model_type} ---") # Added model type for clarity
    print(f"Prompt: {prompt_text}")
    print(f"Specific Banned Words: {specific_banned_words}")
    print(f"Random Ban Percentage: {random_ban_percentage*100:.2f}%")
    print(f"Max New Tokens: {max_new_tokens}")

    all_banned_token_ids = []
    if specific_banned_words:
        # print(f"Tokenizing specific banned words: {specific_banned_words}") # Less verbose
        temp_specific_ids = []
        for word in specific_banned_words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                print(f"Warning: Word '{word}' tokenized to an empty list.")
                continue
            # print(f"  '{word}' -> {token_ids}") # Less verbose
            temp_specific_ids.extend(token_ids)
        if temp_specific_ids:
             print(f"Specific banned words tokenized to IDs: {list(set(temp_specific_ids))[:10]}... (truncated)") # Show some unique IDs
        all_banned_token_ids.extend(temp_specific_ids)

    if random_ban_percentage > 0.0:
        vocab_size = tokenizer.vocab_size
        num_tokens_to_ban = int(random_ban_percentage * vocab_size)
        current_banned_set = set(all_banned_token_ids)
        population_for_random = [i for i in range(vocab_size) if i not in current_banned_set]
        num_tokens_to_ban = min(num_tokens_to_ban, len(population_for_random))

        if num_tokens_to_ban > 0:
            print(f"Randomly banning {num_tokens_to_ban} tokens from {len(population_for_random)} eligible tokens.")
            randomly_banned_ids = random.sample(population_for_random, num_tokens_to_ban)
            all_banned_token_ids.extend(randomly_banned_ids)
        # else: # Less verbose
            # print("No tokens to ban randomly.")

    if all_banned_token_ids:
        all_banned_token_ids = sorted(list(set(all_banned_token_ids)))
        print(f"Final list of {len(all_banned_token_ids)} unique token ID(s) to ban: {all_banned_token_ids[:10]}... (truncated)")
    # else: # Less verbose
        # print("No tokens will be banned for this generation.")

    logits_processors = []
    if all_banned_token_ids:
        processor = TokenBanningLogitsProcessor(banned_token_ids=all_banned_token_ids)
        logits_processors.append(processor)
        # print("TokenBanningLogitsProcessor instantiated.") # Less verbose
    # else: # Less verbose
        # print("No banning processor needed.")

    # print("Encoding prompt...") # Less verbose
    inputs = tokenizer(prompt_text, return_tensors='pt')
    input_ids = inputs.input_ids

    if torch.cuda.is_available() and model.device.type == 'cuda':
        device = model.device
        input_ids = input_ids.to(device)
        # print(f"Input tensors moved to device: {device}") # Less verbose

    # print("Generating response with model.generate()...") # Less verbose
    output_sequences = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processors if logits_processors else None,
    )
    # print("Generation complete.") # Less verbose

    # print("Decoding response...") # Less verbose
    generated_ids = output_sequences[0, input_ids.shape[-1]:]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Generated text: {decoded_text}")
    print(f"--- End of Generation ---")
    return decoded_text

# --- Main Execution Block ---
if __name__ == "__main__":
    ALL_MODEL_IDS = [
        "gpt2-large",
        "google/gemma-2b-it",
        "Qwen/Qwen2-1.5B-Instruct"
        # Add more models here as needed
    ]

    for model_id_str in ALL_MODEL_IDS:
        print(f"\n\n===== Testing Model: {model_id_str} =====")

        model = None
        tokenizer = None

        try:
            print(f"Loading tokenizer for {model_id_str}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id_str)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set tokenizer.pad_token to tokenizer.eos_token for {model_id_str}")

            print(f"Loading model {model_id_str}...")
            model_args = {}
            if "gemma" in model_id_str.lower():
                # model_args['torch_dtype'] = torch.bfloat16 # This can cause issues on some local setups if not supported
                print("Note: For Gemma, in a compatible environment (like some Colab setups), consider torch_dtype=torch.bfloat16 for memory efficiency.")
            if "qwen" in model_id_str.lower():
                model_args['trust_remote_code'] = True
                print("Note: For Qwen models, trust_remote_code=True is set.")

            model = AutoModelForCausalLM.from_pretrained(model_id_str, **model_args)

            if torch.cuda.is_available():
                print(f"Moving model {model_id_str} to CUDA device.")
                model.to('cuda')
            else:
                print(f"CUDA not available for {model_id_str}, using CPU.")

            # Test 1: No bans
            print(f"\n--- Test: No Bans ({model_id_str}) ---")
            prompt1 = "The capital of France is"
            output1 = generate_response(model, tokenizer, prompt1, max_new_tokens=10)
            print(f"Output (No bans, {model_id_str}): {prompt1}{output1}\n")

            # Test 2: Specific word ban
            print(f"\n--- Test: Specific Ban ({model_id_str}) ---")
            prompt2 = "Describe an orange. An orange is a"
            banned_words2 = ["orange", "fruit", "citrus", "round", "sweet"]
            output2 = generate_response(model, tokenizer, prompt2, specific_banned_words=banned_words2, max_new_tokens=30)
            print(f"Output (Banned: {banned_words2}, {model_id_str}): {prompt2}{output2}\n")

            # Test 3: Random percentage ban
            print(f"\n--- Test: Random Ban ({model_id_str}) ---")
            prompt3 = "Tell me an extremely short story about a robot that explores a new planet."
            output3 = generate_response(model, tokenizer, prompt3, random_ban_percentage=0.05, max_new_tokens=50)
            print(f"Output (Random 5% ban, {model_id_str}): {prompt3}{output3}\n")

        except Exception as e:
            print(f"Error during testing of model {model_id_str}: {e}")
            # Potentially log the error or skip to the next model

        finally:
            # Memory Management
            print(f"Finished testing {model_id_str}. Releasing model and tokenizer from memory.")
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # gc.collect() # Good practice, especially in loops like this in Colab

        print(f"===== Finished Model: {model_id_str} =====")

    print("\n\n--- All models tested. Main script execution finished. ---")

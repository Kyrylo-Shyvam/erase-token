# CELL BREAK
# Cell 1: Installs and Imports
!pip install transformers torch sentencepiece accelerate

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import gc # For explicit garbage collection

# CELL BREAK
# Cell 2: TokenBanningLogitsProcessor Class Definition
class TokenBanningLogitsProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids: list[int]):
        super().__init__()
        self.banned_token_ids = banned_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id in self.banned_token_ids:
            if 0 <= token_id < scores.shape[-1]: # Check if token_id is within the vocabulary range
                scores[:, token_id] = -float('inf')
        return scores

# CELL BREAK
# Cell 3: generate_response Function Definition
def generate_response(model, tokenizer, prompt_text: str, specific_banned_words: list[str] = None, random_ban_percentage: float = 0.0, max_new_tokens: int = 50):
    if specific_banned_words is None:
        specific_banned_words = []

    print(f"--- Generating Response for model: {model.config._name_or_path} ---") # Use _name_or_path for better model ID
    print(f"Prompt: {prompt_text}")
    print(f"Specific Banned Words: {specific_banned_words}")
    print(f"Random Ban Percentage: {random_ban_percentage*100:.2f}%")
    print(f"Max New Tokens: {max_new_tokens}")

    all_banned_token_ids = []
    if specific_banned_words:
        print(f"Tokenizing specific banned words: {specific_banned_words}")
        temp_specific_ids = []
        for word in specific_banned_words:
            # Tokenize the word itself. add_special_tokens=False is important.
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids: # Handle cases where a word might tokenize to nothing
                print(f"Warning: Word '{word}' tokenized to an empty list.")
                continue
            print(f"  '{word}' -> {token_ids}")
            temp_specific_ids.extend(token_ids)
        if temp_specific_ids:
             print(f"Specific banned words tokenized to {len(set(temp_specific_ids))} unique ID(s): {list(set(temp_specific_ids))[:20]}... (truncated)")
        all_banned_token_ids.extend(temp_specific_ids)

    if random_ban_percentage > 0.0:
        vocab_size = tokenizer.vocab_size
        num_tokens_to_ban = int(random_ban_percentage * vocab_size)

        current_banned_set = set(all_banned_token_ids)
        population_for_random = [i for i in range(vocab_size) if i not in current_banned_set]

        num_tokens_to_ban = min(num_tokens_to_ban, len(population_for_random))

        if num_tokens_to_ban > 0:
            print(f"Randomly banning {num_tokens_to_ban} tokens from {len(population_for_random)} eligible tokens (vocab size: {vocab_size}).")
            randomly_banned_ids = random.sample(population_for_random, num_tokens_to_ban)
            all_banned_token_ids.extend(randomly_banned_ids)
        else:
            print("No tokens to ban randomly (either percentage too low, vocab too small, or all eligible tokens already banned specifically).")

    if all_banned_token_ids:
        all_banned_token_ids = sorted(list(set(all_banned_token_ids))) # Ensure uniqueness and sort
        print(f"Final list of {len(all_banned_token_ids)} unique token ID(s) to ban: {all_banned_token_ids[:20]}... (truncated if >20)")
    else:
        print("No tokens will be banned for this generation.")

    logits_processors = []
    if all_banned_token_ids:
        processor = TokenBanningLogitsProcessor(banned_token_ids=all_banned_token_ids)
        logits_processors.append(processor)
        print("TokenBanningLogitsProcessor instantiated.")
    else:
        print("No banning processor needed.")

    print("Encoding prompt...")
    inputs = tokenizer(prompt_text, return_tensors='pt')
    input_ids = inputs.input_ids

    # Move inputs to the same device as the model
    if model.device.type == 'cuda': # Check if model is on GPU
        device = model.device
        input_ids = input_ids.to(device)
        print(f"Input tensors moved to device: {device}")

    print("Generating response with model.generate()...")
    output_sequences = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processors if logits_processors else None,
        pad_token_id=tokenizer.pad_token_id # Explicitly set pad_token_id
        # Other parameters like temperature, top_k can be added here
        # no_repeat_ngram_size=2 # Example to prevent some repetition
    )
    print("Generation complete.")

    print("Decoding response...")
    # Slice the output_sequences to get only the generated tokens
    generated_ids = output_sequences[0, input_ids.shape[-1]:]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Generated text: {decoded_text}")
    print(f"--- End of Generation ---")
    return decoded_text

# CELL BREAK
# Cell 4: Main Execution Logic (Model Iteration and Testing)
ALL_MODEL_IDS = [
    "gpt2-large",
    "google/gemma-2b-it",
    "Qwen/Qwen2-1.5B-Instruct"
    # For more models, ensure you have enough RAM/quota in Colab.
    # "meta-llama/Meta-Llama-3-8B-Instruct" # Example: Needs access approval and significant resources
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
            # For Gemma on Colab, torch_dtype might be necessary if running into OOM with default float32
            # model_args['torch_dtype'] = torch.bfloat16 # or torch.float16
            # model_args['low_cpu_mem_usage'] = True # Can also help
            print("Note: For Gemma, if memory issues arise, consider adding to from_pretrained: torch_dtype=torch.bfloat16, low_cpu_mem_usage=True")
        if "qwen" in model_id_str.lower():
            model_args['trust_remote_code'] = True
            print("Note: For Qwen models, trust_remote_code=True is set.")
        # For very large models like Llama 3 8B, you might need:
        # if "llama-3-8b" in model_id_str.lower():
        # model_args['torch_dtype'] = torch.float16 # or bfloat16
        # model_args['device_map'] = 'auto' # for multi-GPU or CPU offload
        # print("Note: For Llama 3 8B, device_map='auto' and torch_dtype might be needed.")

        model = AutoModelForCausalLM.from_pretrained(model_id_str, **model_args)

        if torch.cuda.is_available():
            print(f"Moving model {model_id_str} to CUDA device.")
            model.to('cuda')
        else:
            print(f"CUDA not available for {model_id_str}, using CPU. This might be very slow for large models.")

        # Test 1: No bans
        print(f"\n--- Test: No Bans ({model_id_str}) ---")
        prompt1 = "The capital of France is"
        output1 = generate_response(model, tokenizer, prompt1, max_new_tokens=15) # Increased tokens slightly
        print(f"Output (No bans, {model_id_str}): {prompt1}{output1}\n")

        # Test 2: Specific word ban
        print(f"\n--- Test: Specific Ban ({model_id_str}) ---")
        prompt2 = "Describe an orange. An orange is typically a" # Slightly modified prompt
        # Broader list of banned words to test robustness
        banned_words2 = ["orange", "fruit", "citrus", "round", "sweet", "juicy", "color", "peel"]
        output2 = generate_response(model, tokenizer, prompt2, specific_banned_words=banned_words2, max_new_tokens=35) # Increased tokens
        print(f"Output (Banned: {banned_words2}, {model_id_str}): {prompt2}{output2}\n")

        # Test 3: Random percentage ban
        print(f"\n--- Test: Random Ban ({model_id_str}) ---")
        prompt3 = "Tell me an extremely short story about a robot that explores a new planet and finds a surprising artifact." # More detailed prompt
        output3 = generate_response(model, tokenizer, prompt3, random_ban_percentage=0.05, max_new_tokens=60) # Increased tokens
        print(f"Output (Random 5% ban, {model_id_str}): {prompt3}{output3}\n")

    except Exception as e:
        print(f"Error during testing of model {model_id_str}: {e}")
        if "out of memory" in str(e).lower():
            print("CUDA out of memory. Try restarting the Colab session and using a smaller model or model_args like torch_dtype=torch.float16.")
        # You could add more specific error handling here

    finally:
        # Memory Management: Crucial in Colab
        print(f"Finished testing {model_id_str}. Releasing model and tokenizer from memory.")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() # Force garbage collection

    print(f"===== Finished Model: {model_id_str} =====")

print("\n\n--- All models tested. Main script execution finished. ---")
print("Reminder: If you faced CUDA out-of-memory errors, try selecting a GPU with more RAM (e.g., A100, V100 in Colab Pro) or use smaller models/quantization.")

# CELL BREAK
# Cell 5: Example of using generate_response directly (Optional)
# print("\n\n===== Manual Test with generate_response (using last loaded model configuration if loop failed, or re-load a model) =====")
# This cell is for users to play around with the generate_response function directly.
# They would need to ensure a model and tokenizer are loaded.
# For example, uncomment and modify the following lines:

# if 'tokenizer' not in locals() or 'model' not in locals():
#     print("Loading a default model for manual testing (e.g., gpt2)...")
#     manual_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     if manual_tokenizer.pad_token is None:
#         manual_tokenizer.pad_token = manual_tokenizer.eos_token
#     manual_model = AutoModelForCausalLM.from_pretrained("gpt2")
#     if torch.cuda.is_available():
#         manual_model.to('cuda')
# else:
#     print(f"Using the last configured tokenizer and model ({tokenizer.name_or_path}) for manual testing.")
#     manual_tokenizer = tokenizer
#     manual_model = model # This might be problematic if the loop failed and model is None

# my_prompt = "The best way to learn programming is"
# my_banned_words = ["code", "computer", "online"]
# my_output = generate_response(manual_model, manual_tokenizer, my_prompt, specific_banned_words=my_banned_words, random_ban_percentage=0.01, max_new_tokens=50)
# print(f"Manual Test Output: {my_prompt}{my_output}")

# print("\n--- Manual testing cell finished ---")
# Note: The above manual test cell is largely commented out to prevent auto-execution
# if the main loop fails and `model`/`tokenizer` are not in a good state.
# Users should uncomment and adapt it carefully.

# END OF CONTENT

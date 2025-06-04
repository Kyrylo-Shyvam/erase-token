import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import random
# import gc # Will be used in Colab

# --- TokenBanningLogitsProcessor Definition (Updated for Attempt Tracking) ---
class TokenBanningLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 globally_banned_token_ids: list[int] = None,
                 banned_sequences_ids: list[list[int]] = None,
                 tokenizer=None, # New parameter for decoding attempts
                 attempt_log_list: list = None): # New parameter for logging
        super().__init__()
        self.globally_banned_token_ids = globally_banned_token_ids if globally_banned_token_ids is not None else []
        self.banned_sequences_ids = banned_sequences_ids if banned_sequences_ids is not None else []
        self.tokenizer = tokenizer
        self.attempt_log_list = attempt_log_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        original_scores_clone = None
        if self.attempt_log_list is not None and self.tokenizer is not None:
            original_scores_clone = scores.clone()

        # Apply global bans first
        for token_id in self.globally_banned_token_ids:
            if 0 <= token_id < scores.shape[-1]:
                scores[:, token_id] = -float('inf')

        # Apply sequence bans
        if self.banned_sequences_ids:
            # Assuming effective batch_size=1 for this logic within generate context
            # input_ids shape is (batch_size, sequence_length)
            # scores shape is (batch_size, vocab_size)
            for i in range(input_ids.shape[0]): # Iterate over batch if batch_size > 1
                current_sequence_tensor = input_ids[i]
                current_seq_len = current_sequence_tensor.shape[0]

                for seq_to_ban in self.banned_sequences_ids:
                    if not seq_to_ban or len(seq_to_ban) < 2:
                        continue

                    token_to_suppress_if_match = seq_to_ban[-1]
                    prefix_to_match = seq_to_ban[:-1]
                    len_prefix = len(prefix_to_match)

                    if current_seq_len >= len_prefix:
                        current_suffix_tensor = current_sequence_tensor[-len_prefix:]
                        current_suffix_list = current_suffix_tensor.tolist()

                        if current_suffix_list == prefix_to_match:
                            if 0 <= token_to_suppress_if_match < scores.shape[-1]:
                                scores[i, token_to_suppress_if_match] = -float('inf')

        # Log attempted banned tokens (after all bans applied)
        if original_scores_clone is not None: # Implies self.attempt_log_list and self.tokenizer are not None
            # Process for batch_size=1 for simplicity in logging.
            # In a multi-beam scenario, input_ids might have batch_size > 1.
            # This logging will only reflect the first item in such a batch.
            # For standard greedy/sample, batch_size is typically 1.

            # We are interested in the top candidate for the first sequence in the batch
            # before any bans were applied for *this specific generation step*.
            top_candidate_token_id = torch.argmax(original_scores_clone[0]).item()

            # Check if this top candidate was actually banned in the modified scores
            if 0 <= top_candidate_token_id < scores.shape[-1] and \
               scores[0, top_candidate_token_id] == -float('inf'):

                ban_type_str = "unknown" # Default
                is_global_ban = top_candidate_token_id in self.globally_banned_token_ids
                is_sequence_ban = False

                # Check if it was a sequence ban for the first item in the batch
                # This re-checks the condition that led to the ban for logging purposes.
                current_sequence_tensor_log = input_ids[0]
                current_seq_len_log = current_sequence_tensor_log.shape[0]
                for seq_to_ban_log in self.banned_sequences_ids:
                    if not seq_to_ban_log or len(seq_to_ban_log) < 2:
                        continue
                    if top_candidate_token_id == seq_to_ban_log[-1]:
                        prefix_to_match_log = seq_to_ban_log[:-1]
                        len_prefix_log = len(prefix_to_match_log)
                        if current_seq_len_log >= len_prefix_log and \
                           current_sequence_tensor_log[-len_prefix_log:].tolist() == prefix_to_match_log:
                            is_sequence_ban = True
                            break

                if is_global_ban and is_sequence_ban:
                    ban_type_str = "global_and_sequence"
                elif is_global_ban:
                    ban_type_str = "global"
                elif is_sequence_ban:
                    ban_type_str = "sequence"

                attempt_info = {
                    'step': input_ids.shape[1], # Current length of input_ids (generation step)
                    'attempted_token_id': top_candidate_token_id,
                    'attempted_token_str': self.tokenizer.decode([top_candidate_token_id]),
                    'ban_type': ban_type_str
                }
                self.attempt_log_list.append(attempt_info)

        return scores

# --- Generate Response Function (Updated for Attempt Tracking) ---
def generate_response(model, tokenizer, prompt_text: str,
                      specific_banned_words: list[str] = None,
                      random_ban_percentage: float = 0.0,
                      banned_phrases_for_sequence_ban: list[str] = None,
                      max_new_tokens: int = 50):
    if specific_banned_words is None: specific_banned_words = []
    if banned_phrases_for_sequence_ban is None: banned_phrases_for_sequence_ban = []

    print(f"--- Generating Response for model: {model.config._name_or_path if hasattr(model, 'config') and hasattr(model.config, '_name_or_path') else 'N/A'} ---")
    print(f"Prompt: {prompt_text}")
    print(f"Globally Banned Words: {specific_banned_words}")
    print(f"Banned Phrases for Sequence Ban: {banned_phrases_for_sequence_ban}")
    print(f"Random Ban Percentage: {random_ban_percentage*100:.2f}%")
    print(f"Max New Tokens: {max_new_tokens}")

    all_banned_token_ids = []
    tokenized_banned_sequences = []
    generation_attempt_log = [] # Initialize log list

    if specific_banned_words:
        # ... (rest of specific word banning logic remains the same) ...
        print(f"Tokenizing specific words for global banning: {specific_banned_words}")
        temp_specific_ids = []
        for word in specific_banned_words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if not token_ids:
                print(f"Warning: Word '{word}' tokenized to an empty list for global ban.")
                continue
            # print(f"  '{word}' -> {token_ids} (for global ban)") # Reduced verbosity
            temp_specific_ids.extend(token_ids)
        if temp_specific_ids:
            all_banned_token_ids.extend(list(set(temp_specific_ids)))

    if banned_phrases_for_sequence_ban:
        # ... (rest of phrase banning logic remains the same) ...
        print(f"Tokenizing phrases for sequence banning: {banned_phrases_for_sequence_ban}")
        for phrase in banned_phrases_for_sequence_ban:
            token_ids = tokenizer.encode(phrase, add_special_tokens=False)
            if len(token_ids) >= 2:
                tokenized_banned_sequences.append(token_ids)
                # print(f"  '{phrase}' -> {token_ids} (for sequence ban)") # Reduced verbosity
            elif token_ids:
                # print(f"  '{phrase}' -> {token_ids} (single token from phrase, adding to global ban)") # Reduced verbosity
                all_banned_token_ids.extend(token_ids)
            else:
                print(f"Warning: Phrase '{phrase}' tokenized to an empty list for sequence ban.")

    if random_ban_percentage > 0.0:
        # ... (rest of random banning logic remains the same) ...
        vocab_size = tokenizer.vocab_size
        current_globally_banned_set = set(all_banned_token_ids)
        num_tokens_to_ban_randomly = int(random_ban_percentage * vocab_size)
        population_for_random = [i for i in range(vocab_size) if i not in current_globally_banned_set]
        num_tokens_to_ban_randomly = min(num_tokens_to_ban_randomly, len(population_for_random))
        if num_tokens_to_ban_randomly > 0:
            # print(f"Randomly selecting {num_tokens_to_ban_randomly} additional tokens for global banning from {len(population_for_random)} eligible tokens.") # Reduced verbosity
            randomly_banned_ids = random.sample(population_for_random, num_tokens_to_ban_randomly)
            all_banned_token_ids.extend(randomly_banned_ids)

    if all_banned_token_ids:
        all_banned_token_ids = sorted(list(set(all_banned_token_ids)))
        print(f"Final list of {len(all_banned_token_ids)} unique token ID(s) for global ban: {all_banned_token_ids[:10]}... (truncated)")
    # else: print("No tokens will be globally banned for this generation.")

    if tokenized_banned_sequences:
        print(f"Final list of {len(tokenized_banned_sequences)} sequences for sequence ban: {tokenized_banned_sequences[:5]}... (truncated)")
    # else: print("No sequences will be banned for this generation.")

    logits_processors = []
    if all_banned_token_ids or tokenized_banned_sequences:
        processor = TokenBanningLogitsProcessor(
            globally_banned_token_ids=all_banned_token_ids,
            banned_sequences_ids=tokenized_banned_sequences,
            tokenizer=tokenizer, # Pass tokenizer for decoding in logger
            attempt_log_list=generation_attempt_log # Pass log list
        )
        logits_processors.append(processor)
        print("TokenBanningLogitsProcessor instantiated with attempt tracking.")
    # else: print("No banning processor needed.")

    # print("Encoding prompt...") # Reduced verbosity
    inputs = tokenizer(prompt_text, return_tensors='pt')
    input_ids = inputs.input_ids

    if torch.cuda.is_available() and model.device.type == 'cuda':
        device = model.device
        input_ids = input_ids.to(device)

    # print("Generating response with model.generate()...") # Reduced verbosity
    output_sequences = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processors if logits_processors else None,
        pad_token_id=tokenizer.pad_token_id
    )
    # print("Generation complete.") # Reduced verbosity

    # print("Decoding response...") # Reduced verbosity
    generated_ids = output_sequences[0, input_ids.shape[-1]:]
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Generated text: {decoded_text}")
    print(f"--- End of Generation ---")
    return decoded_text, generation_attempt_log # Return log

# --- Main Execution Block ---
if __name__ == "__main__":
    ALL_MODEL_IDS = [
        "gpt2-large",
    ]

    for model_id_str in ALL_MODEL_IDS:
        print(f"\n\n===== Testing Model with Attempt Tracking: {model_id_str} =====")

        model = None
        tokenizer = None

        try:
            print(f"Loading tokenizer for {model_id_str}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id_str)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            # ... (model loading args and model loading as before) ...
            model_args = {}
            if "gemma" in model_id_str.lower(): model_args['torch_dtype'] = torch.float16
            if "qwen" in model_id_str.lower(): model_args['trust_remote_code'] = True
            model = AutoModelForCausalLM.from_pretrained(model_id_str, **model_args)
            if torch.cuda.is_available(): model.to('cuda')


            # Test 4: Sequence Banning Test (Focus for logging)
            print(f"\n--- Test: Sequence Banning with Attempt Tracking ({model_id_str}) ---")
            prompt4 = "I really like to visit New York City. New York is a great place. I love New York!"
            globally_banned_words4 = ["great", "place"]
            banned_phrases_for_seq_ban4 = ["New York"]

            decoded_output4, attempts4 = generate_response( # Capture attempts
                model,
                tokenizer,
                prompt4,
                specific_banned_words=globally_banned_words4,
                banned_phrases_for_sequence_ban=banned_phrases_for_seq_ban4,
                max_new_tokens=30
            )
            # Updated print for Test 4
            print(f"Output (Globally Banned: {globally_banned_words4}, Sequence Banned: {banned_phrases_for_seq_ban4}, Model: {model_id_str}):\nPrompt: {prompt4}\nGenerated: {decoded_output4}")
            if attempts4:
                print("Banned token generation attempts:")
                for attempt in attempts4:
                    print(f"  Step {attempt['step']}: Tried to generate '{attempt['attempted_token_str']}' (ID: {attempt['attempted_token_id']}), banned due to: {attempt['ban_type']}")
            else:
                print("No banned token generation attempts logged.")
            print("\n")


            # Test 5: Sequence Banning with a common phrase (Focus for logging)
            print(f"\n--- Test: Sequence Banning Common Phrase with Attempt Tracking ({model_id_str}) ---")
            prompt5 = "The quick brown fox jumps over the lazy dog. I saw a brown fox today."
            banned_phrases_for_seq_ban5 = ["brown fox"]

            decoded_output5, attempts5 = generate_response( # Capture attempts
                model,
                tokenizer,
                prompt5,
                banned_phrases_for_sequence_ban=banned_phrases_for_seq_ban5,
                max_new_tokens=25
            )
            # Updated print for Test 5
            print(f"Output (Sequence Banned: {banned_phrases_for_seq_ban5}, Model: {model_id_str}):\nPrompt: {prompt5}\nGenerated: {decoded_output5}")
            if attempts5:
                print("Banned token generation attempts:")
                for attempt in attempts5:
                    print(f"  Step {attempt['step']}: Tried to generate '{attempt['attempted_token_str']}' (ID: {attempt['attempted_token_id']}), banned due to: {attempt['ban_type']}")
            else:
                print("No banned token generation attempts logged.")
            print("\n")

        except Exception as e:
            print(f"Error during testing of model {model_id_str}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # ... (cleanup as before) ...
            print(f"Finished testing {model_id_str}. Releasing model and tokenizer from memory.")
            if 'model' in locals() and model is not None : del model
            if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n\n--- All models tested (or selected models for focused test). Main script execution finished. ---")

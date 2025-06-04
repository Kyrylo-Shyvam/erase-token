# CELL BREAK
# Scenario 1: Baseline Example - Describe an Apple
# CELL: Demonstration Scenario - Describe an Apple (Banning "apple")

print("======================================================================")
print("== Demonstration: Describe an Apple (Banning 'apple') ==")
print("======================================================================")

model_id_for_demo = "gpt2-large"
prompt_demo = "Describe an apple."
specific_banned_words_demo = ["apple", "apples"] # Ban singular and plural
random_ban_percentage_demo = 0.0
max_new_tokens_demo = 60

model_demo = None
tokenizer_demo = None

try:
    # --- Actual Loading logic for this cell ---
    print(f"DEMO: Loading tokenizer for {model_id_for_demo}...")
    tokenizer_demo = AutoTokenizer.from_pretrained(model_id_for_demo)
    if tokenizer_demo.pad_token is None:
        tokenizer_demo.pad_token = tokenizer_demo.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token for {model_id_for_demo}")

    model_args_demo = {}
    if "gemma" in model_id_for_demo.lower():
        model_args_demo['torch_dtype'] = torch.float16
        print(f"Note: For Gemma ({model_id_for_demo}), torch_dtype=torch.float16 is set.")
    if "qwen" in model_id_for_demo.lower():
        model_args_demo['trust_remote_code'] = True
        print(f"Note: For Qwen ({model_id_for_demo}), trust_remote_code=True is set.")

    print(f"DEMO: Loading model {model_id_for_demo} with args {model_args_demo}...")
    model_demo = AutoModelForCausalLM.from_pretrained(model_id_for_demo, **model_args_demo)

    if torch.cuda.is_available():
        print(f"Moving model {model_id_for_demo} to CUDA device.")
        model_demo.to('cuda')
    else:
        print(f"CUDA not available for {model_id_for_demo}, using CPU.")
    # --- END Actual Loading logic ---

    print(f"\n--- Scenario Details ---")
    print(f"Model: {model_id_for_demo}")
    print(f"Prompt: {prompt_demo}")
    print(f"Max New Tokens: {max_new_tokens_demo}")

    print(f"\n--- Generating Baseline (No Bans) ---")
    baseline_output_demo = generate_response(
        model_demo,
        tokenizer_demo,
        prompt_demo,
        max_new_tokens=max_new_tokens_demo
    )
    print(f"\nBaseline Output:\n{prompt_demo}{baseline_output_demo}")

    print(f"\n--- Generating with Bans (Banned: {specific_banned_words_demo}) ---")
    banned_output_demo = generate_response(
        model_demo,
        tokenizer_demo,
        prompt_demo,
        specific_banned_words=specific_banned_words_demo,
        random_ban_percentage=random_ban_percentage_demo,
        max_new_tokens=max_new_tokens_demo
    )
    print(f"\nOutput with Bans:\n{prompt_demo}{banned_output_demo}")

except Exception as e:
    print(f"Error during demo scenario '{prompt_demo[:30]}...': {e}")
finally:
    print("\nCleaning up resources for demo scenario...")
    if 'model_demo' in locals() and model_demo is not None: del model_demo
    if 'tokenizer_demo' in locals() and tokenizer_demo is not None: del tokenizer_demo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Demo scenario finished and cleaned up.")
print("======================================================================")


# CELL BREAK
# Scenario 2: Baseline Example - Math Problem
# CELL: Demonstration Scenario - Math Problem (Banning "+")

print("======================================================================")
print("== Demonstration: Math Problem (Banning '+') ==")
print("======================================================================")

model_id_for_demo = "gpt2-large"
prompt_demo = "What is 2 + 2?"
specific_banned_words_demo = ["+"] # Ban the plus symbol
random_ban_percentage_demo = 0.0
max_new_tokens_demo = 10

model_demo = None
tokenizer_demo = None

try:
    # --- Actual Loading logic for this cell ---
    print(f"DEMO: Loading tokenizer for {model_id_for_demo}...")
    tokenizer_demo = AutoTokenizer.from_pretrained(model_id_for_demo)
    if tokenizer_demo.pad_token is None:
        tokenizer_demo.pad_token = tokenizer_demo.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token for {model_id_for_demo}")

    model_args_demo = {}
    if "gemma" in model_id_for_demo.lower():
        model_args_demo['torch_dtype'] = torch.float16
        print(f"Note: For Gemma ({model_id_for_demo}), torch_dtype=torch.float16 is set.")
    if "qwen" in model_id_for_demo.lower():
        model_args_demo['trust_remote_code'] = True
        print(f"Note: For Qwen ({model_id_for_demo}), trust_remote_code=True is set.")

    print(f"DEMO: Loading model {model_id_for_demo} with args {model_args_demo}...")
    model_demo = AutoModelForCausalLM.from_pretrained(model_id_for_demo, **model_args_demo)

    if torch.cuda.is_available():
        print(f"Moving model {model_id_for_demo} to CUDA device.")
        model_demo.to('cuda')
    else:
        print(f"CUDA not available for {model_id_for_demo}, using CPU.")
    # --- END Actual Loading logic ---

    print(f"\n--- Scenario Details ---")
    print(f"Model: {model_id_for_demo}")
    print(f"Prompt: {prompt_demo}")
    print(f"Specific Banned Words: {specific_banned_words_demo}")
    print(f"Max New Tokens: {max_new_tokens_demo}")

    print(f"\n--- Generating Baseline (No Bans) ---")
    baseline_output_demo = generate_response(
        model_demo,
        tokenizer_demo,
        prompt_demo,
        max_new_tokens=max_new_tokens_demo
    )
    print(f"\nBaseline Output:\n{prompt_demo}{baseline_output_demo}")

    print(f"\n--- Generating with Bans (Banned: {specific_banned_words_demo}) ---")
    banned_output_demo = generate_response(
        model_demo,
        tokenizer_demo,
        prompt_demo,
        specific_banned_words=specific_banned_words_demo,
        random_ban_percentage=random_ban_percentage_demo,
        max_new_tokens=max_new_tokens_demo
    )
    print(f"\nOutput with Bans:\n{prompt_demo}{banned_output_demo}")

except Exception as e:
    print(f"Error during demo scenario '{prompt_demo[:30]}...': {e}")
finally:
    print("\nCleaning up resources for demo scenario...")
    if 'model_demo' in locals() and model_demo is not None: del model_demo
    if 'tokenizer_demo' in locals() and tokenizer_demo is not None: del tokenizer_demo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Demo scenario finished and cleaned up.")
print("======================================================================")


# CELL BREAK
# Scenario 3: "Awareness" Test - Fruits
# CELL: Demonstration Scenario - "Awareness" Test (Fruits)

print("======================================================================")
print("== Demonstration: 'Awareness' Test - Name Ten Fruits (Ban 'apple') ==")
print("======================================================================")

# Consider using a more capable model if gpt2-large struggles too much with listing.
# model_id_for_demo = "google/gemma-2b-it"
model_id_for_demo = "gpt2-large"
prompt_demo = "Name ten different types of fruits."
specific_banned_words_demo = ["apple", "apples"]
random_ban_percentage_demo = 0.0
max_new_tokens_demo = 75 # Allow more tokens for a list

model_demo = None
tokenizer_demo = None

try:
    # --- Actual Loading logic for this cell ---
    print(f"DEMO: Loading tokenizer for {model_id_for_demo}...")
    tokenizer_demo = AutoTokenizer.from_pretrained(model_id_for_demo)
    if tokenizer_demo.pad_token is None:
        tokenizer_demo.pad_token = tokenizer_demo.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token for {model_id_for_demo}")

    model_args_demo = {}
    if "gemma" in model_id_for_demo.lower():
        model_args_demo['torch_dtype'] = torch.float16
        print(f"Note: For Gemma ({model_id_for_demo}), torch_dtype=torch.float16 is set.")
    if "qwen" in model_id_for_demo.lower():
        model_args_demo['trust_remote_code'] = True
        print(f"Note: For Qwen ({model_id_for_demo}), trust_remote_code=True is set.")

    print(f"DEMO: Loading model {model_id_for_demo} with args {model_args_demo}...")
    model_demo = AutoModelForCausalLM.from_pretrained(model_id_for_demo, **model_args_demo)

    if torch.cuda.is_available():
        print(f"Moving model {model_id_for_demo} to CUDA device.")
        model_demo.to('cuda')
    else:
        print(f"CUDA not available for {model_id_for_demo}, using CPU.")
    # --- END Actual Loading logic ---

    print(f"\n--- Scenario Details ---")
    print(f"Model: {model_id_for_demo}")
    print(f"Prompt: {prompt_demo}")
    print(f"Specific Banned Words: {specific_banned_words_demo}")
    print(f"Max New Tokens: {max_new_tokens_demo}")

    print(f"\n--- Generating Baseline (No Bans) ---")
    baseline_output_demo = generate_response(
        model_demo,
        tokenizer_demo,
        prompt_demo,
        max_new_tokens=max_new_tokens_demo
    )
    print(f"\nBaseline Output:\n{prompt_demo}{baseline_output_demo}")

    print(f"\n--- Generating with Bans (Banned: {specific_banned_words_demo}) ---")
    banned_output_demo = generate_response(
        model_demo,
        tokenizer_demo,
        prompt_demo,
        specific_banned_words=specific_banned_words_demo,
        random_ban_percentage=random_ban_percentage_demo,
        max_new_tokens=max_new_tokens_demo
    )
    print(f"\nOutput with Bans:\n{prompt_demo}{banned_output_demo}")
    print("\nAnalyze: Is 'apple' (and 'apples') omitted? Does the model struggle or list fewer valid fruits?")

except Exception as e:
    print(f"Error during demo scenario '{prompt_demo[:30]}...': {e}")
finally:
    print("\nCleaning up resources for demo scenario...")
    if 'model_demo' in locals() and model_demo is not None: del model_demo
    if 'tokenizer_demo' in locals() and tokenizer_demo is not None: del tokenizer_demo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Demo scenario finished and cleaned up.")
print("======================================================================")


# CELL BREAK
# Scenario 4: Random Ban Test - Short Story
# CELL: Demonstration Scenario - Random Ban Test (Short Story)

print("======================================================================")
print("== Demonstration: Random Ban Test - Short Story ==")
print("======================================================================")

model_id_for_demo = "gpt2-large"
prompt_demo = "Tell me a short story about a friendly robot who discovered a hidden garden."
max_new_tokens_demo = 150
# Test a range of random ban percentages, including 0% as a baseline
random_ban_percentages_to_test = [0.0, 0.01, 0.05, 0.10, 0.20]

model_demo = None
tokenizer_demo = None

try:
    # --- Actual Loading logic for this cell ---
    print(f"DEMO: Loading tokenizer for {model_id_for_demo}...")
    tokenizer_demo = AutoTokenizer.from_pretrained(model_id_for_demo)
    if tokenizer_demo.pad_token is None:
        tokenizer_demo.pad_token = tokenizer_demo.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token for {model_id_for_demo}")

    model_args_demo = {}
    if "gemma" in model_id_for_demo.lower():
        model_args_demo['torch_dtype'] = torch.float16
        print(f"Note: For Gemma ({model_id_for_demo}), torch_dtype=torch.float16 is set.")
    if "qwen" in model_id_for_demo.lower():
        model_args_demo['trust_remote_code'] = True
        print(f"Note: For Qwen ({model_id_for_demo}), trust_remote_code=True is set.")

    print(f"DEMO: Loading model {model_id_for_demo} with args {model_args_demo}...")
    model_demo = AutoModelForCausalLM.from_pretrained(model_id_for_demo, **model_args_demo)

    if torch.cuda.is_available():
        print(f"Moving model {model_id_for_demo} to CUDA device.")
        model_demo.to('cuda')
    else:
        print(f"CUDA not available for {model_id_for_demo}, using CPU.")
    # --- END Actual Loading logic ---

    print(f"\n--- Scenario Details ---")
    print(f"Model: {model_id_for_demo}")
    print(f"Prompt: {prompt_demo}")
    print(f"Max New Tokens: {max_new_tokens_demo}")

    for pc_idx, percentage in enumerate(random_ban_percentages_to_test):
        print(f"\n--- Test {pc_idx+1}: Random Ban Percentage: {percentage*100:.2f}% ---")
        # For random ban, specific_banned_words is None or empty list
        output_demo = generate_response(
            model_demo,
            tokenizer_demo,
            prompt_demo,
            specific_banned_words=[], # Ensure no specific bans for this test
            random_ban_percentage=percentage,
            max_new_tokens=max_new_tokens_demo
        )
        print(f"Output (Random Ban {percentage*100:.2f}%):\n{prompt_demo}{output_demo}")
        if percentage > 0.05: # Arbitrary threshold for analysis prompt
             print("\nAnalyze: Observe output degradation, coherence, use of unusual words, or other notable changes as random ban % increases.")

except Exception as e:
    print(f"Error during demo scenario '{prompt_demo[:30]}...': {e}")
finally:
    print("\nCleaning up resources for demo scenario...")
    if 'model_demo' in locals() and model_demo is not None: del model_demo
    if 'tokenizer_demo' in locals() and tokenizer_demo is not None: del tokenizer_demo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Demo scenario finished and cleaned up.")
print("======================================================================")

# END OF CONTENT

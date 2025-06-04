# CELL BREAK
# CELL: User Experimentation Playground

print("=======================================================================")
print("== User Experimentation Playground ==")
print("== Modify the parameters below and re-run this cell to experiment. ==")
print("=======================================================================")

# --- 1. Configure Your Experiment ---

# Choose a model ID from the list:
# Available: "gpt2-large", "google/gemma-2b-it", "Qwen/Qwen2-1.5B-Instruct"
# (Make sure the model you select was loaded in the main execution cell, or adapt to load here)
# For simplicity, this cell will attempt to load the selected model.
# Be mindful of Colab's VRAM limits when switching models frequently.
selected_model_id = "gpt2-large"  # @param ["gpt2-large", "google/gemma-2b-it", "Qwen/Qwen2-1.5B-Instruct"]

user_prompt_text = "Describe a beautiful sunset over the ocean."  # @param {type:"string"}

# List of words/phrases to ban specifically. Example: ["beautiful", "ocean"]
user_specific_banned_words = []  # @param {type:"raw"}

# Percentage of random vocabulary to ban (0.0 to 1.0). Example: 0.05 for 5%
user_random_ban_percentage = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.01}

user_max_new_tokens = 100  # @param {type:"integer"}

# --- 2. Setup and Run Experiment ---

# (Helper function to load model and tokenizer if not already loaded)
# Note: This is a simplified loader for this cell.
# The main loop (Cell 4) has more robust loading for multiple models.
# Consider running Cell 4 first if you encounter issues here or want all models pre-downloaded.

_loaded_models_cache = {} # Basic cache for this cell

def get_model_and_tokenizer_for_experiment(model_id):
    if model_id in _loaded_models_cache:
        print(f"Using cached model and tokenizer for {model_id}.")
        return _loaded_models_cache[model_id]['model'], _loaded_models_cache[model_id]['tokenizer']

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token for {model_id}")

    model_args = {}
    if "gemma" in model_id.lower():
        # For Colab, bfloat16 is often good for Gemma 2B if on T4 GPU or newer
        # V100 might prefer float16. A100 supports bfloat16.
        # Using float16 as a safer default if bfloat16 is not universally available/performant
        model_args['torch_dtype'] = torch.float16
        # model_args['low_cpu_mem_usage'] = True # Already in main loop, good for Colab
        print(f"Note: For Gemma ({model_id}), torch_dtype=torch.float16 is set for this cell.")
    if "qwen" in model_id.lower():
        model_args['trust_remote_code'] = True
        print(f"Note: For Qwen ({model_id}), trust_remote_code=True is set.")

    print(f"Loading model {model_id} with args {model_args}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_args)
        if torch.cuda.is_available():
            print(f"Moving model {model_id} to CUDA device.")
            model.to('cuda')
        else:
            print(f"CUDA not available for {model_id}, using CPU.")

        # Clear cache if it grows too large (e.g., more than 1 model)
        # Only keep one model loaded via this cell's cache
        if len(_loaded_models_cache) >= 1:
            print("Clearing previous model(s) from this cell's cache to save memory...")
            # Basic FIFO like, or just clear all but current
            # This doesn't handle models loaded by other cells.
            # Make a copy of keys to iterate over for deletion
            keys_to_delete = [key for key in _loaded_models_cache.keys() if key != model_id]

            for key_to_del in keys_to_delete:
                print(f"Removing {key_to_del} from this cell's cache.")
                # It's important to ensure that model and tokenizer are actually objects before del
                if 'model' in _loaded_models_cache[key_to_del] and _loaded_models_cache[key_to_del]['model'] is not None:
                    del _loaded_models_cache[key_to_del]['model']
                if 'tokenizer' in _loaded_models_cache[key_to_del] and _loaded_models_cache[key_to_del]['tokenizer'] is not None:
                    del _loaded_models_cache[key_to_del]['tokenizer']
                del _loaded_models_cache[key_to_del] # remove entry

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        _loaded_models_cache[model_id] = {'model': model, 'tokenizer': tokenizer}
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        print("Please ensure you have enough VRAM/RAM, and the model ID is correct.")
        print("You might need to restart the Colab session if memory issues persist.")
        return None, None

current_model, current_tokenizer = get_model_and_tokenizer_for_experiment(selected_model_id)

if current_model and current_tokenizer:
    print(f"\n--- Experiment Details ---")
    print(f"Model: {selected_model_id}") # Or use current_model.config._name_or_path
    print(f"Prompt: {user_prompt_text}")
    print(f"Max New Tokens: {user_max_new_tokens}")

    # Baseline Generation (No Bans)
    print(f"\n--- Generating Baseline (No Bans) ---")
    baseline_output = generate_response(
        model=current_model,
        tokenizer=current_tokenizer,
        prompt_text=user_prompt_text,
        max_new_tokens=user_max_new_tokens
    )
    print(f"\nBaseline Output:\n{user_prompt_text}{baseline_output}")

    # Generation with User-Specified Bans
    if user_specific_banned_words or user_random_ban_percentage > 0.0:
        print(f"\n--- Generating with User Bans ---")
        # print(f"Specific Banned Words: {user_specific_banned_words}") # generate_response will print this
        # print(f"Random Ban Percentage: {user_random_ban_percentage*100:.2f}%") # generate_response will print this

        banned_output = generate_response(
            model=current_model,
            tokenizer=current_tokenizer,
            prompt_text=user_prompt_text,
            specific_banned_words=user_specific_banned_words,
            random_ban_percentage=user_random_ban_percentage,
            max_new_tokens=user_max_new_tokens
        )
        print(f"\nOutput with Bans:\n{user_prompt_text}{banned_output}")
    else:
        print("\nNo bans specified by the user (specific or random). Skipping ban generation.")

    print("\n=======================================================================")
    print("== Experiment Finished. Modify parameters above and re-run. ==")
    print("=======================================================================")
else:
    print("Could not run experiment due to model loading issues. Please check errors above.")
    print("Ensure the selected model ID is correct and Colab has sufficient resources.")
# END OF CELL CONTENT

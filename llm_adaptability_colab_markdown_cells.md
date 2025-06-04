# CELL BREAK
# Markdown Cell 1: Introduction and Initial Setup
# (To be placed near the beginning of the notebook, after the title perhaps, before the first code cell)

# Investigating LLM Adaptability via Dynamic Token Suppression

**Objective:** This notebook allows experimentation with the response generation of various small, open-source LLMs when specific tokens or a random percentage of the vocabulary are dynamically banned.

## How to Use This Notebook

1.  **Select a GPU Runtime:**
    *   For best performance and to run the larger models, ensure you have a GPU runtime selected.
    *   Go to **Runtime -> Change runtime type**.
    *   Select **T4 GPU** (or other available GPU like L4, V100, A100) from the "Hardware accelerator" dropdown.
    *   *Note: Free Colab tier GPUs (like T4) have VRAM limitations (approx 15-16GB). Be mindful of model sizes.*

2.  **Run Cells Sequentially (Recommended):**
    *   **Cell 1 (Installs and Imports):** Run this first to install necessary libraries (`transformers`, `torch`, etc.) and import base modules. This may take a minute.
    *   **Cell 2 (TokenBanningLogitsProcessor):** Defines the custom logic for token banning.
    *   **Cell 3 (generate_response Function):** Defines the main function for generating responses with banning.
    *   **Cell 4 (Main Execution Logic - Automated Tests):** This cell iterates through the pre-defined models (`gpt2-large`, `google/gemma-2b-it`, `Qwen/Qwen2-1.5B-Instruct`) and runs a series of automated tests.
        *   *Model Downloads:* The first time you run this cell (or load a new model), the models will be downloaded from Hugging Face. This can take several minutes and consume disk space.
        *   *Memory Management:* This cell attempts to load one model at a time and clean up memory afterwards.
    *   **Cell 5 (User Experimentation Playground):** This is where you can define your own experiments!
        *   Select a model, write a prompt, choose words to ban, and set a random ban percentage.
        *   This cell also manages loading and unloading the selected model for your experiment.
    *   **Demonstration Scenario Cells (Following the Playground):** These cells provide pre-configured examples of different banning scenarios. Run them individually to see specific effects.

3.  **Running "Run All":** You can use **Runtime -> Run all**, but be patient, especially the first time, due to model downloads and sequential testing. If you encounter memory issues, try running cells one by one after a runtime restart.

## Output Interpretation
*   For each experiment, the notebook will typically show:
    *   The model used.
    *   The input prompt.
    *   Banning parameters.
    *   Baseline output (without bans).
    *   Output with the specified bans.
*   Pay attention to how the model adapts (or fails to adapt) to the token bans.

# CELL BREAK
# Markdown Cell 2: Memory Management Advice
# (To be placed after the "User Experimentation Playground" cell, or near the end of the notebook)

## Understanding and Managing Memory (VRAM/RAM)

Large Language Models are memory-intensive. Google Colab provides limited resources, especially VRAM (GPU memory) on the free tier.

**How this Notebook Manages Memory:**
*   **Automatic Cleanup:** Most cells that load models (the main test loop, user playground, demo scenarios) include code to `del` the model and tokenizer objects and then call `torch.cuda.empty_cache()` and `gc.collect()`. This helps free up VRAM and RAM after a model is no longer immediately needed.
*   **User Playground Cache:** The "User Experimentation Playground" cell attempts to keep only one model loaded at a time *within its own execution context*. If you select a new model in that cell, it tries to unload the previous one.

**Tips for Users:**
*   **Out-Of-Memory (OOM) Errors:** If you see errors like `CUDA out of memory`, it means the GPU ran out of VRAM.
    1.  **Restart Session:** The most effective solution is often **Runtime -> Restart session**. This clears all memory. You'll need to re-run cells from the beginning (installs, definitions).
    2.  **Run Cells Individually:** After a restart, run cells one by one, especially the model-loading and generation cells. This gives memory a chance to clear between heavy operations.
    3.  **Stick to One Model:** If you're experimenting heavily in the "User Playground," try to stick with one model for a while before switching, or ensure previous models are fully cleared.
    4.  **Model Sizes:**
        *   `gpt2-large` is relatively smaller.
        *   `google/gemma-2b-it` (approx 5GB in float32, less in float16/bfloat16) and `Qwen/Qwen2-1.5B-Instruct` (approx 3GB) are larger. Loading multiple such models *simultaneously* will likely cause OOM errors. This notebook is designed to load them sequentially.
*   **`torch_dtype`:** For models like Gemma, using `torch_dtype=torch.float16` or `torch_dtype=torch.bfloat16` (if supported by your Colab GPU, e.g., T4, L4, A100 for bfloat16) can significantly reduce VRAM usage compared to `float32`. The notebook attempts to use `torch.float16` for Gemma in user-facing cells as a safer default for wider Colab GPU compatibility.

**If issues persist, you might be hitting the fundamental limits of the available Colab resources for the chosen models.**

# CELL BREAK
# Markdown Cell 3: Model Specific Notes (Optional but Recommended)
# (Can be near the memory management advice or part of initial setup)

## Model Specific Notes & Considerations

*   **`google/gemma-2b-it`**:
    *   This is an instruction-tuned version of Gemma 2B.
    *   Requires `sentencepiece` for its tokenizer (installed in the first cell).
    *   Benefits significantly from `torch_dtype=torch.bfloat16` (on compatible GPUs like T4, L4, A100) or `torch_dtype=torch.float16` for memory savings. This notebook defaults to `torch.float16` in interactive cells for wider Colab GPU compatibility.
*   **`Qwen/Qwen2-1.5B-Instruct`**:
    *   This model may require `trust_remote_code=True` when loading, as its implementation might involve custom code from its Hugging Face repository. The notebook includes this argument when loading Qwen models.
*   **Hugging Face Model Cards:** For any model, always refer to its Hugging Face model card for the most up-to_date information on loading, usage, and potential requirements.

This notebook is pre-configured for the models listed. If you wish to experiment with other Hugging Face models, you may need to:
1.  Add the model identifier to relevant lists (e.g., in the `User Experimentation Playground` dropdown if you customize it, or in the main test loop in Cell 4).
2.  Check if it requires `trust_remote_code=True` or other specific loading arguments.
3.  Be very mindful of its size and your Colab instance's VRAM.
# END OF CONTENT

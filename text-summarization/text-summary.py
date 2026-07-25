import torch
# AutoTokenizer encodes text into the numerical tokens the model expects.
# AutoModelForSeq2SeqLM loads a sequence-to-sequence model like DistilBART.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# ---------------------------
# 1. Load model + tokenizer once
# ---------------------------
# Loading outside the function means we download and initialize
# the model only at startup, not on every prediction.
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Tokenizer: converts raw strings into token IDs the model can process.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Model: the actual summarization model.
# - torch_dtype=torch.bfloat16 uses less memory on GPUs that support it.
# - device_map="..." automatically puts the model on GPU if available,
#   otherwise keeps it on CPU.
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)


# ---------------------------
# 2. Define the summarization logic
# ---------------------------
def summarize(text: str) -> str:
    """Summarize the given text using the loaded model."""
    # Guard against empty input - return a friendly message instead of failing.
    if not text.strip():
        return "Please enter some text to summarize."

    # Convert text to model inputs.
    # - truncation=True: cuts input if it's too long for the model.
    # - max_length=1024: maximum tokens the model can handle.
    # - return_tensors="pt": returns PyTorch tensors.
    # - .to(model.device): moves tensors to the same device (CPU/GPU) as the model.
    inputs = tokenizer(text, truncation=True, return_tensors="pt", max_length=1024).to(
        model.device
    )

    # Generate the summary.
    # - input_ids: the tokenized input text.
    # - num_beams=4: uses beam search (keeps 4 best sequences) for better quality.
    # - min_length / max_length: control the summary size.
    # - early_stopping=True: stop generation once all beams reach end-of-sequence.
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=30,
        max_length=130,
        early_stopping=True,
    )

    # Convert generated token IDs back to a readable string.
    # - skip_special_tokens=True: removes <pad>, <s>, </s> tokens.
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# ---------------------------
# 3. Build the web UI with Gradio
# ---------------------------
# Gradio creates a web interface around the summarize() function.
demo = gr.Interface(
    fn=summarize,  # The Python function to call.
    inputs=gr.Textbox(
        lines=10,           # Multi-line text box height.
        placeholder="Paste your text here...",
        label="Input Text",
    ),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer",
    description="Summarize long text using DistilBART (CNN/DailyMail).",
    # examples lets users click a sample text to try instantly.
    examples=[
        [
            "Artificial intelligence is transforming many industries by automating tasks, improving decision-making, and enabling new products and services."
        ]
    ],
)

# ---------------------------
# 4. Start the web server
# ---------------------------
# __name__ == "__main__" ensures the server only runs when this file
# is executed directly (not when imported as a module).
if __name__ == "__main__":
    demo.launch()

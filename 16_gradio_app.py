import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import (
    TextIteratorStreamer,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread
import os


## Model Download
# from unsloth import FastLanguageModel
# max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model_id = 'universalml0/NepaliGPT-pretrained-llama3.1-v0-small'
# model_id = 'universalml/NepaliGPT-2.0'

# hf_token = os.environ['HF_TOKEN_2']
hf_token = os.environ['HF_TOKEN']
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_id,
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     token = hf_token, # use one if using gated models like meta-llama/Llama-2-7b-hf
# )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = 'auto',
    torch_dtype=torch.bfloat16, # if newer gpu: bfloat16
    # https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention
    # attn_implementation="flash_attention_2", # Works with llama model
    token = hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token = hf_token )

# Our Prompt
alpaca_prompt = """ {}

### Instruction:
{}

### Input:
{}

### Response:
{}"""



## Gradio
def respond(
    message,
    history: list[tuple[str, str]],
    language,
    input,
    max_tokens,
    temperature,
    top_p,
    num_beams,
):
  if language == "English":
    system_message = "Below is instruction that describes a task, paired with input which provide further context. Write a response  that appropriately complete the request."
  if language == "Nepali" :
    system_message = "Below is instruction in Nepali or English language that describes a task, paired with input which provide further context. Write a response in Nepali language that appropriately complete the request ."
  # chat_history = ""
  # if len(history)>0:
  #   chat_history = f"user: {history[-1][0]}, assistant:{history[-1][1]}"
  # FastLanguageModel.for_inference(model)
  inputs = tokenizer([
        alpaca_prompt.format(
            system_message, # system
            message, # instruction
            "", # input
            # chat_history, # chat history
            "", # output - leave this blank for generation!
        ),
    ], return_tensors = "pt").to("cuda")
  streamer = TextIteratorStreamer(
          tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True
  )
  generate_kwargs = dict(
          **inputs,
          max_new_tokens=max_tokens,
          streamer=streamer,
          # num_beams=num_beams, # streamer cannot use beam search yet
          temperature=temperature,
          top_p=top_p,
          # repetition_penalty=1.0,
      )
  def generate_and_signal_complete():
    model.generate(**generate_kwargs)

  t1 = Thread(target=generate_and_signal_complete)
  t1.start()

  partial_text = ""
  for new_text in streamer:
    partial_text += new_text
    yield partial_text

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Radio(["English", "Nepali"],value='English', label="language(response)", info="In which language, you want your response to be in? For translation type question for either to Nepali or English, I recommended you to choose English language here irrespective of translation to English or to Nepali"),
        # gr.Textbox(value="Below is instruction in Nepali or English language that describes a task, paired with input which provide further context. Write a response in Nepali language that appropriately complete the request .", label="System message"),
        gr.Textbox(value="", label="Input",info="For further context of your question."),
        gr.Slider(minimum=1, maximum=5124, value=1024, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Slider(minimum=1, maximum=10, value=4, step=1, label="number of beams", info="Sorry Streamer can't use beam search yet. Choose higher for better result but with slow generation"),
    ],
    title="NepaliGPT🇳🇵",
    description = "Disclaimer:Lower the Temperature below for more precise and exact answer like in language translation, grammer fixing etc question, may be (0.1 to 0.3). Increase the temperature for more creative answer (may lead to soft error), (may be 0.6 - 1.2). You can always retry if the response isnot of your liking.",
)


if __name__ == "__main__":
    demo.launch()
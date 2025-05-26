import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

chat_history_ids = None

def respond(user_input, history):
    global chat_history_ids
    if user_input.lower() == "quit":
        chat_history_ids = None
        return "", []

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    history = history + [(user_input, response)]
    return "", history

with gr.Blocks() as chatbot_ui:
    gr.Markdown("## AI Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...")
    clear = gr.Button("Clear")

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: (None, []), None, [msg, chatbot])

chatbot_ui.launch()

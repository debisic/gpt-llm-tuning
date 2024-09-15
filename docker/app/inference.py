""" Building the /generate/ methods of the API to get inference"""

import torch
from transformers import TextStreamer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def create_prompt(instruction: str) -> str:
    """ Creating prompt template """

    prompt_template = """
    Find below the instructions to complete a task. Write an answer that completes conveniently the request.
    ### Instruction:
    [INSTRUCTION]

    ### Response:
    """
    return prompt_template .replace("[INSTRUCTION]", instruction)


# print(create_prompt("Que sais tu d'Enedis?"))

def stream(user_prompt,tokenizer, model):
    """ Making prediction with the mdeol """
    # generate_prompt
    prompt = create_prompt(user_prompt)
    #print(f"prompt: {prompt}")

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    output_ids = model.generate(**inputs, streamer=streamer, max_new_tokens=500)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def prediction(prompt):

    """Loading model and tokenizer for processing and then make prediction"""
    model_name = "./gpt2-fine-tuned_merged"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    return stream(prompt,tokenizer, model)

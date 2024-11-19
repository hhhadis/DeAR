import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

openai.api_base = "your_api_base"
openai.api_key = "your_api_key"
LLAMA_MODEL_PATH = "your-llama2-7b-path"  
CHATGLM_MODEL_PATH = "your-chatglm3-6b-path"  

def chatgpt_inference(prompt_text):
        rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
        )
        response_txt = rsp.choices[0].message.content.strip()
        return response_txt

def llama_inference(prompt_text):
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH)
    model = model.eval()  
    if torch.cuda.is_available():
       model = model.to("cuda") 
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()} 
    with torch.no_grad():  
        outputs = model.generate(
            inputs["input_ids"],
            max_length=256, 
            temperature=0.7,  
            top_p=0.9, 
            do_sample=True,  
            pad_token_id=tokenizer.eos_token_id 
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def chatglm_inference(prompt_text):
    tokenizer = AutoTokenizer.from_pretrained(CHATGLM_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(CHATGLM_MODEL_PATH, trust_remote_code=True)
    model = model.eval()  
    if torch.cuda.is_available():
        model = model.half().cuda()
    with torch.no_grad():  
        response, _ = model.chat(tokenizer, prompt_text, history=[])  
    return response
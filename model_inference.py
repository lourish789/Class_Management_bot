from transformers import pipeline

generator = pipeline("text-generation", model="./teacher-coach-lora", tokenizer="./teacher-coach-lora")

def get_coach_response(message):
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    response = generator(prompt, max_new_tokens=100, do_sample=True)[0]['generated_text']
    return response.split("### Response:\n")[-1].strip()

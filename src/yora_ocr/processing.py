from llama_cpp import Llama
from .helper import purchase_model_to_json

output_format = purchase_model_to_json()
print(output_format)

llm = Llama.from_pretrained(
    repo_id="lmstudio-community/Qwen3.5-0.8B-GGUF",
    filename="*Q8_0.gguf",
)

output = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You extract receipt data and return it only in JSON format following the given structure.",
        },
        {"role": "user", "content": "AldiBanane 1,99Tomate 1,99Apfel 1,50Gesamt 5,48"},
    ],
    response_format={"type": "json_object", "schema": output_format},
)

print(output)

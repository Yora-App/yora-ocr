from llama_cpp import Llama
from .helper import model_to_json
from .schema import Purchase

output_format = model_to_json(Purchase)
print(output_format)

llm = Llama.from_pretrained(
    repo_id="lmstudio-community/Qwen3.5-0.8B-GGUF",
    filename="*Q8_0.gguf",
    n_gpu_layers = -1,
    verbose = True
)

output = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You extract receipt data and return it only in JSON format following the given structure.",
        },
        {
            "role": "user",
            "content": "ALDI SOD"
            "GalileistraBe 14"
            "69115 Heidelberg"
            "EUR"
            "230334 Champ. WeiB 400g"
            "1,99A"
            "231308 Spitzpap. Rot 500g"
            "0,97A"
            "229425 Zucchini lose"
            "0,488 kg x"
            "1,99 EUR/kg"
            "K-U-N-D-E-N-B-E-L-E-G"
            "Kartenzahlung girocard"
            "Betrag"
            "33,73 EUR"
            "05.01.2026"
            "10:56"
            "Zahlung erfolgt"
            "sum"
            "33,73",
        },
    ],
    response_format={"type": "json_object", "schema": output_format},
)

print(output)

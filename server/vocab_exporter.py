from transformers import AutoTokenizer
import json
import os

def export_vocab():
    print("Exporting Vocab...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab = tokenizer.get_vocab()
    
    # Invert: ID -> Token
    id2token = {v: k for k, v in vocab.items()}
    
    # Save
    os.makedirs("web", exist_ok=True)
    with open("web/vocab.json", "w") as f:
        json.dump(id2token, f)
    
    print("Saved web/vocab.json")

if __name__ == "__main__":
    export_vocab()

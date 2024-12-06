from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route("/")
def home():
    return render_template("index.html")

def load_dynamic_context():
    # Load context from a text file or database
    with open("data/context.txt", "r") as file:
        return file.read()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    context = load_dynamic_context()
    prompt = f"question: {user_input}  context: {context}"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    # outputs = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
    outputs = model.generate(
        inputs, 
        max_length=50, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,  # Balanced randomness
        top_p=0.85,       # Diverse token sampling
        do_sample=True,   # Enable non-deterministic sampling
        no_repeat_ngram_size=3,  # Avoid repeated phrases
        repetition_penalty=1.2,  # Penalize exact repetition
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": answer})

if __name__ == "__main__":
    print("\033[92m" + "Server is running successfully on http://127.0.0.1:5001" + "\033[0m")
    app.run(debug=True, port=5001)
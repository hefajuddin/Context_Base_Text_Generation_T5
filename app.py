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
        temperature=0.7, 
        top_k=50, 
        top_p=0.9,
        repetition_penalty=1.2  # Penalizes repeating tokens
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
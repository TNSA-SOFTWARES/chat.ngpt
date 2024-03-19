from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import re

app = Flask(__name__)

# Load the pre-trained model architecture
model_json_file = 'text_generation_model.json'

if not os.path.exists(model_json_file):
    print("Model architecture file not found.")
    exit()

try:
    with open(model_json_file, 'r') as json_file:
        model_architecture = json.load(json_file)

    input_dim = model_architecture['input_dim']
    output_dim = model_architecture['output_dim']
    word_to_index = model_architecture['word_to_index']
    index_to_word = model_architecture['index_to_word']

except Exception as e:
    print("Error loading model architecture:", e)
    exit()

# Load sentences from file and preprocess them
data_file = 'sentences.txt'
if not os.path.exists(data_file):
    print("Sentences file not found.")
    exit()

with open(data_file, 'r', encoding='utf-8') as sentences_file:
    sentences = [line.strip() for line in sentences_file.readlines()]

# Initialize random weights and biases for text generation
weights_hidden = np.random.randn(input_dim, 1024)
biases_hidden = np.zeros(128)
weights_output = np.random.randn(1024, output_dim)
biases_output = np.zeros(output_dim)

# Define a function to generate text
def generate_text(seed_text, next_words, model_architecture, temperature=1.0):
    # Access word_to_index and index_to_word dictionaries from model_architecture
    word_to_index = model_architecture['word_to_index']
    index_to_word = model_architecture['index_to_word']
    # Rest of the function remains the same...

# Define a function to answer queries
def answer_query(query, data, model_architecture):
    # Rest of the function remains the same...

# Define a function to preprocess user input
def preprocess_user_input(user_input):
    # Remove special characters and numbers
    user_input = re.sub(r'[^a-zA-Z\s]', '', user_input)
    # Convert to lowercase
    user_input = user_input.lower()
    return user_input.strip()

# Render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Handle user queries
@app.route('/query', methods=['POST'])
def query():
    try:
        user_input = request.form['query']
        
        # Preprocess user input
        user_input = preprocess_user_input(user_input)
        
        # Fill in the blanks in the user's query
        filled_query = fill_in_the_blanks(user_input, model_architecture, sentences, temperature=1.7)

        # Answer the filled query using the modified function
        response = answer_query(filled_query, sentences, model_architecture)

        return jsonify({'success': True, 'message': response})

    except Exception as e:
        print("Error handling query:", e)
        return jsonify({'success': False, 'message': 'Error handling query'})

if __name__ == '__main__':
    app.run(debug=True)

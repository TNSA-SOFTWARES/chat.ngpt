from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import json
import re

app = Flask(__name__)

# Load the pre-trained model architecture
model_json_file = 'text_generation_model.json'
data_file = 'data.txt'

if not os.path.exists(model_json_file) or not os.path.exists(data_file):
    print("Model architecture file or data file not found.")
    exit()

with open(model_json_file, 'r') as json_file:
    model_architecture = json.load(json_file)

input_dim = model_architecture['input_dim']
output_dim = model_architecture['output_dim']
word_to_index = model_architecture['word_to_index']
index_to_word = model_architecture['index_to_word']

# Read data from the file with explicit encoding specification
with open(data_file, 'r', encoding='utf-8') as data_file:
    data = data_file.read().splitlines()

# Initialize random weights and biases if the weight files are not available
try:
    weights_hidden = np.loadtxt('weights_hidden.csv', delimiter=',')
    biases_hidden = np.loadtxt('biases_hidden.csv', delimiter=',')
    weights_output = np.loadtxt('weights_output.csv', delimiter=',')
    biases_output = np.loadtxt('biases_output.csv', delimiter=',')
except OSError:
    # Initialize random weights if files are not found
    weights_hidden = np.random.randn(input_dim, 1024)
    biases_hidden = np.zeros(128)
    weights_output = np.random.randn(1024, output_dim)
    biases_output = np.zeros(output_dim)

# Define a function to fill in the blanks
def fill_in_the_blanks(query, model_architecture, data, temperature=1.0):
    # Find all occurrences of blanks in the query
    matches = re.finditer(r'________*', query)
    
    # Iterate through matches and replace blanks with predicted words
    for match in matches:
        start, end = match.span()
        blank_size = end - start
        seed_text = query[:start].strip()  # Use the text before the blank as context
        
        # Generate text based on the blank size
        if seed_text in data:
            generated_text = seed_text
        else:
            generated_text = generate_text(seed_text, next_words=blank_size, model_architecture=model_architecture, temperature=temperature)

        # Replace the blank with the generated text
        query = query[:start] + generated_text + query[end:]

    return query

# Define a function to generate text
def generate_text(seed_text, next_words, model_architecture, temperature=1.0):
    generated_text = seed_text
    recent_words = seed_text.split()  # Store the most recent words

    for _ in range(next_words):
        # Check if the most recent word is in the vocabulary
        if recent_words[-1] in word_to_index:
            # Rest of the code remains the same
            token_list = generated_text.split()
            token_list = token_list[-(input_dim - 1):]

            token_indices = [word_to_index[word] for word in token_list]
            token_encoding = np.zeros(input_dim)

            for idx in token_indices:
                token_encoding[idx] = 1

            hidden_layer_input = np.dot(token_encoding, weights_hidden) + biases_hidden
            hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))

            output_layer_input = np.dot(hidden_layer_output, weights_output) + biases_output
            output_layer_output = np.exp(output_layer_input) / np.sum(np.exp(output_layer_input))

            scaled_output = np.log(output_layer_output) / temperature
            scaled_output = np.exp(scaled_output - np.max(scaled_output))
            scaled_output = scaled_output / scaled_output.sum()

            # Filter the next word based on the coherence with preceding words
            filtered_output = scaled_output.copy()
            for idx in range(output_dim):
                word = index_to_word[str(idx)]
                if word not in recent_words and not word.isdigit():  # Exclude numeric tokens
                    filtered_output[idx] = 0

            # Normalize the filtered output probabilities
            filtered_output /= filtered_output.sum()

            next_word_index = np.random.choice(range(output_dim), p=filtered_output)
            next_word = index_to_word[str(next_word_index)]

            if next_word not in recent_words and not next_word.isdigit():  # Exclude numeric tokens
                recent_words.append(next_word)
                if len(recent_words) > input_dim - 1:
                    recent_words.pop(0)

                generated_text += " " + next_word
        else:
            break

    return generated_text

# Define a function to answer queries
def answer_query(query, data, model_architecture):
    # Check if the query is in the data
    if query in data:
        return data[data.index(query) + 1]  # Return the answer after the query
    else:
        # If not in data, generate a response using the model
        generated_response = generate_text(query, next_words=500, model_architecture=model_architecture, temperature=1.7)
        return generated_response

# Accept user input and generate a response
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_input = request.form['query']
    
    # Fill in the blanks in the user's query
    filled_query = fill_in_the_blanks(user_input, model_architecture, data, temperature=1.7)

    # Answer the filled query using the modified function
    response = answer_query(filled_query, data, model_architecture)

    # Return the response
    response_data = {
        'success': True,
        'message': response,
    }
    return jsonify(response_data)

# Print message indicating the NGEN was just updated
print("NGEN was just updated.")

if __name__ == '__main__':
    app.run(debug=True)

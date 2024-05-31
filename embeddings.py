from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to generate embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Use .squeeze() to convert to 2D array if necessary
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


def generate_embeddings_open_ai(
    user_input=None,
):
    response = openai_client.embeddings.create(
                model= "text-embedding-ada-002",
                input=user_input
            )
    return response.data[0].embedding


def generate_text(user_input):
    response = openai_client.chat.completions.create(
                model= "gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": "answer only based on the provided context, if you dont find and answer there, return, I dont know."},
                    {"role": "user", "content": user_input},
                ]
            )
    return response.choices[0].message.content




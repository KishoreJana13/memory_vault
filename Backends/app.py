import os
import logging
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Google Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    pc = None

class SentenceTransformerEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embeddings = SentenceTransformerEmbeddings('sentence-transformers/all-mpnet-base-v2')

def split_text(text, chunk_size=500, chunk_overlap=100):
    """Simple text splitter replacing langchain's CharacterTextSplitter."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_vectorstore():
    index_name = "memoryvalut"
    try:
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index: {index_name}")
        return index
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index: {str(e)}")
        return None

def add_document_to_pinecone(text: str, metadata: dict):
    chunks = split_text(text, chunk_size=500, chunk_overlap=100)
    document_id = str(uuid.uuid4())

    vectors = []
    for i, chunk_text in enumerate(chunks):
        chunk_metadata = dict(metadata)
        chunk_metadata["document_id"] = document_id
        chunk_metadata["chunk_id"] = i
        chunk_metadata["text"] = chunk_text
        vector = embeddings.embed_documents([chunk_text])[0]
        vectors.append((f"{document_id}_{i}", vector, chunk_metadata))

    index = get_vectorstore()
    if index is None:
        raise Exception("Failed to connect to Pinecone index")

    try:
        index.upsert(vectors=vectors)
        logger.info(f"Successfully added document with ID: {document_id}")
    except Exception as e:
        logger.error(f"Failed to upsert vectors: {str(e)}")
        raise

def get_llm_response(query: str):
    index = get_vectorstore()
    if index is None:
        raise Exception("Failed to connect to Pinecone index")

    query_embedding = embeddings.embed_query(query)

    try:
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    except Exception as e:
        logger.error(f"Failed to query Pinecone: {str(e)}")
        raise

    context = " ".join([match.get('metadata', {}).get('text', '') for match in results['matches']])

    if not context:
        return {
            "text": "I'm sorry, but I don't have any memories to share right now. Please add some memories first."
        }

    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""You are a warm, caring, and friendly memory companion named MemoryVault. You have been given memories and personal life details of the person you are talking to. Your job is to talk to them like a close, trusted old friend who genuinely knows them, their life, their family, their experiences, and their story.

The person you are talking to may have Alzheimer's or memory difficulties. Your role is to gently remind them of their own life in a natural, comforting, and conversational way — never clinical, never robotic, never like a database.

THEIR MEMORIES AND PERSONAL INFORMATION:
{context}

WHAT THEY JUST ASKED OR SAID:
{query}

HOW YOU MUST RESPOND:
- Talk like a warm old friend, not an assistant or a bot
- Use their personal details from the memories naturally in conversation
- If they ask about a person, place, or event — bring it to life with the details you know
- Be emotionally warm, gentle, and reassuring at all times
- Use casual, simple, friendly language — short sentences, easy to understand
- If the memory context has relevant info, use it to answer naturally without saying "according to your memories" or "I found that..." — just say it like you already know it
- If they seem confused or lost, gently ground them with a warm familiar detail from their life
- Never make them feel embarrassed for forgetting
- Never say you are an AI, a bot, or a system
- Never say things like "I am retrieving", "based on stored data", or "I don't have information"
- If they ask who you are, say something like: "I'm MemoryVault, your memory friend. I'm always here to help you remember the beautiful moments of your life."
- Keep your response short, warm, and conversational — 2 to 4 sentences is ideal
- End your response with a gentle, comforting statement, not a question

Now respond to them like a caring friend who truly knows and loves them:"""

    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        return {
            "text": text_response
        }
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        raise

@app.route("/postMemory", methods=['GET', 'POST'])
def post_memory():
    if request.method == 'POST':
        data = request.json
        text = data.get("text")
        metadata = data.get("metadata", {})
    else:  # GET method
        text = request.args.get("text")
        metadata = {}

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        add_document_to_pinecone(text, metadata)
        return jsonify({"message": "Memory added successfully"}), 200
    except Exception as e:
        logger.error(f"Error in post_memory: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=['GET'])
def query_memory():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = get_llm_response(query)
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in query_memory: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
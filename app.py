import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Core imports
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# GitHub model integration
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# BM25 for hybrid search
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Flask imports
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import threading
import time
from werkzeug.utils import secure_filename
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class SearchResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str

class GitHubModelLLM(LLM):
    """Custom LLM wrapper for GitHub models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", api_token: str = None):
        super().__init__()
        self.model_name = model_name
        self.api_token = api_token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://models.inference.ai.azure.com"
        
    @property
    def _llm_type(self) -> str:
        return "github_model"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the GitHub model API"""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling GitHub model: {str(e)}"

class HybridSearchRetriever:
    """Combines vector search (Qdrant) with BM25 keyword search"""
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, 
                 embeddings_model: SentenceTransformer, redis_client: redis.Redis):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model
        self.redis_client = redis_client
        self.bm25 = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25"""
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def build_bm25_index(self, documents: List[Document]):
        """Build BM25 index from documents"""
        self.documents = documents
        tokenized_docs = [self.preprocess_text(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Cache BM25 index in Redis
        bm25_data = {
            'documents': [doc.page_content for doc in documents],
            'metadata': [doc.metadata for doc in documents]
        }
        self.redis_client.set(f"bm25_index_{self.collection_name}", 
                            json.dumps(bm25_data, default=str))
    
    def vector_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform vector search using Qdrant"""
        query_vector = self.embeddings_model.encode(query).tolist()
        
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for point in search_result:
            results.append(SearchResult(
                content=point.payload.get('content', ''),
                score=point.score,
                metadata=point.payload.get('metadata', {}),
                source='vector'
            ))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        if not self.bm25:
            return []
            
        query_tokens = self.preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                results.append(SearchResult(
                    content=self.documents[idx].page_content,
                    score=scores[idx],
                    metadata=self.documents[idx].metadata,
                    source='keyword'
                ))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[SearchResult]:
        """Combine vector and keyword search results"""
        
        # Get results from both methods
        vector_results = self.vector_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)
        
        # Normalize scores
        def normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
            if not results:
                return results
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            if max_score == min_score:
                return results
            
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
            return results
        
        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)
        
        # Combine and rerank
        combined_results = {}
        
        # Add vector results
        for result in vector_results:
            key = result.content[:100]  # Use first 100 chars as key
            combined_results[key] = SearchResult(
                content=result.content,
                score=result.score * vector_weight,
                metadata=result.metadata,
                source=result.source
            )
        
        # Add keyword results
        for result in keyword_results:
            key = result.content[:100]
            if key in combined_results:
                # Combine scores
                combined_results[key].score += result.score * keyword_weight
                combined_results[key].source = 'hybrid'
            else:
                combined_results[key] = SearchResult(
                    content=result.content,
                    score=result.score * keyword_weight,
                    metadata=result.metadata,
                    source=result.source
                )
        
        # Sort by combined score
        final_results = sorted(combined_results.values(), 
                             key=lambda x: x.score, reverse=True)
        
        return final_results[:top_k]

class ContextAwareChatbot:
    """Main chatbot class with context awareness"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", 
                 redis_url: str = "redis://localhost:6379",
                 github_token: str = None):
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize models
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = GitHubModelLLM(api_token=github_token)
        
        # Initialize components
        self.collection_name = "chatbot_knowledge"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize retriever
        self.retriever = None
        
        # Create collection if it doesn't exist
        self._create_collection()
        
    def _create_collection(self):
        """Create Qdrant collection for storing embeddings"""
        try:
            collections = self.qdrant_client.get_collections()
            if self.collection_name not in [col.name for col in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{}] * len(documents)
            
        # Split documents into chunks
        doc_objects = []
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            for chunk in chunks:
                doc_objects.append(Document(
                    page_content=chunk,
                    metadata={**metadata[i], 'chunk_id': len(doc_objects)}
                ))
        
        # Generate embeddings and store in Qdrant
        points = []
        for i, doc in enumerate(doc_objects):
            embedding = self.embeddings_model.encode(doc.page_content).tolist()
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
            ))
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Initialize hybrid retriever
        self.retriever = HybridSearchRetriever(
            self.qdrant_client,
            self.collection_name,
            self.embeddings_model,
            self.redis_client
        )
        
        # Build BM25 index
        self.retriever.build_bm25_index(doc_objects)
        
        # Cache document count in Redis
        self.redis_client.set(f"doc_count_{self.collection_name}", len(doc_objects))
        
        print(f"Added {len(doc_objects)} document chunks to knowledge base")
    
    def get_conversation_context(self, user_id: str) -> Dict:
        """Get conversation context from Redis"""
        context_key = f"context_{user_id}"
        context = self.redis_client.get(context_key)
        
        if context:
            return json.loads(context)
        else:
            return {
                'session_start': datetime.now().isoformat(),
                'message_count': 0,
                'topics': [],
                'user_preferences': {}
            }
    
    def update_conversation_context(self, user_id: str, context: Dict):
        """Update conversation context in Redis"""
        context_key = f"context_{user_id}"
        self.redis_client.setex(
            context_key,
            3600,  # 1 hour expiry
            json.dumps(context, default=str)
        )
    
    def chat(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """Main chat function with context awareness"""
        
        # Get conversation context
        context = self.get_conversation_context(user_id)
        context['message_count'] += 1
        
        # Perform hybrid search
        if self.retriever:
            search_results = self.retriever.hybrid_search(user_input, top_k=3)
            context_docs = [result.content for result in search_results]
            
            # Create context-aware prompt
            context_prompt = f"""
            Based on the following context and conversation history, provide a helpful response:
            
            Context from knowledge base:
            {chr(10).join(context_docs)}
            
            Current conversation context:
            - Session started: {context['session_start']}
            - Message count: {context['message_count']}
            - Previous topics: {', '.join(context['topics'][-3:])}
            
            User question: {user_input}
            
            Please provide a comprehensive answer based on the context provided.
            """
            
            # Get response from LLM
            response = self.llm(context_prompt)
            
            # Update context
            # Simple topic extraction (can be enhanced with NLP)
            words = user_input.lower().split()
            potential_topics = [word for word in words if len(word) > 4]
            context['topics'].extend(potential_topics[:2])
            context['topics'] = list(set(context['topics']))[-10:]  # Keep last 10 unique topics
            
            # Update conversation context
            self.update_conversation_context(user_id, context)
            
            return {
                'response': response,
                'sources': [{'content': r.content[:200], 'score': r.score, 'source': r.source} 
                           for r in search_results],
                'context': context
            }
        else:
            return {
                'response': "Please add documents to the knowledge base first.",
                'sources': [],
                'context': context
            }
    
    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user"""
        context_key = f"context_{user_id}"
        self.redis_client.delete(context_key)
        self.memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        doc_count = self.redis_client.get(f"doc_count_{self.collection_name}")
        
        return {
            'collection_name': self.collection_name,
            'document_count': int(doc_count) if doc_count else 0,
            'qdrant_status': 'connected',
            'redis_status': 'connected',
            'embedding_model': 'all-MiniLM-L6-v2',
            'llm_model': self.llm.model_name
        }

# Flask Web Application
app = Flask(__name__)
CORS(app)

# Global chatbot instance
chatbot = None

# Initialize chatbot
def initialize_chatbot():
    global chatbot
    chatbot = ContextAwareChatbot(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        github_token=os.getenv("GITHUB_TOKEN")
    )
    
    # Add sample documents
    sample_documents = [
        """
        Python is a high-level programming language known for its simplicity and readability. 
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        Python is widely used in web development, data science, artificial intelligence, and automation.
        """,
        """
        Machine Learning is a subset of artificial intelligence that enables computers to learn 
        and improve from experience without being explicitly programmed. Common types include 
        supervised learning, unsupervised learning, and reinforcement learning.
        """,
        """
        Vector databases are specialized databases designed to store and query high-dimensional vectors.
        They are essential for applications like similarity search, recommendation systems, and 
        semantic search in AI applications.
        """,
        """
        LangChain is a framework for developing applications powered by language models. 
        It provides tools for creating chains of operations, managing memory, and integrating 
        with various data sources and APIs.
        """
    ]
    
    sample_metadata = [
        {"topic": "programming", "category": "python"},
        {"topic": "ai", "category": "machine_learning"},
        {"topic": "databases", "category": "vector_db"},
        {"topic": "frameworks", "category": "langchain"}
    ]
    
    chatbot.add_documents(sample_documents, sample_metadata)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    data = request.get_json()
    user_input = data.get('message', '')
    user_id = data.get('user_id', 'default')
    
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        result = chatbot.chat(user_input, user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats_api():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    try:
        stats = chatbot.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_documents', methods=['POST'])
def add_documents_api():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    data = request.get_json()
    documents = data.get('documents', [])
    metadata = data.get('metadata', [])
    
    if not documents:
        return jsonify({"error": "Documents are required"}), 400
    
    try:
        chatbot.add_documents(documents, metadata)
        return jsonify({"message": "Documents added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history_api():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    data = request.get_json()
    user_id = data.get('user_id', 'default')
    
    try:
        chatbot.clear_conversation_history(user_id)
        return jsonify({"message": "History cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Initialize chatbot in a separate thread
    threading.Thread(target=initialize_chatbot).start()
    
    # Wait for chatbot to initialize
    time.sleep(2)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

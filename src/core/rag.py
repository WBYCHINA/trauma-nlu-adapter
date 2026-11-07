import json
import os

class RAGRetriever:
    def __init__(self, knowledge_base_path: str, embedding_model: str = "bge-small-zh-v1.5"):
        self.knowledge_base = []
        self.use_vector_search = False
        
        if os.path.exists(knowledge_base_path):
            self.load_knowledge_base(knowledge_base_path)
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            self.client = chromadb.PersistentClient(path="data/chroma_db")
            
            self.collection = self.client.get_or_create_collection(
                name="first_aid_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.embedding_model = SentenceTransformer(embedding_model)
            self.use_vector_search = True
            
            self.build_index()
            print("RAG vector search enabled")
        except Exception as e:
            print(f"Failed to initialize vector search: {e}")
            print("Using keyword-based retrieval instead")
    
    def load_knowledge_base(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.knowledge_base = data
    
    def build_index(self):
        if not self.use_vector_search:
            return
        
        if self.collection.count() > 0:
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, guide in enumerate(self.knowledge_base):
            doc_text = f"{guide['injury_type']} {guide['body_location']} {guide['severity']}\n"
            doc_text += "步骤：" + "\n".join(guide['steps']) + "\n"
            doc_text += "注意事项：" + "\n".join(guide['precautions'])
            
            documents.append(doc_text)
            metadatas.append({
                "injury_type": guide['injury_type'],
                "body_location": guide['body_location'],
                "severity": guide['severity']
            })
            ids.append(f"guide_{idx}")
        
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
    
    def retrieve(self, query: str, top_k: int = 3):
        if not self.use_vector_search:
            return []
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        guides = []
        for i in range(top_k):
            if i < len(results['documents'][0]):
                guides.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        
        return guides
    
    def retrieve_by_intent(self, injury_type: str, body_location: str, severity: str):
        for guide in self.knowledge_base:
            if (guide['injury_type'] == injury_type and
                guide['body_location'] == body_location and
                guide['severity'] == severity):
                return guide
        
        for guide in self.knowledge_base:
            if guide['injury_type'] == injury_type:
                return guide
        
        if self.use_vector_search:
            query = f"{injury_type} {body_location} {severity}"
            results = self.retrieve(query, top_k=1)
            if results:
                for guide in self.knowledge_base:
                    if guide['injury_type'] == results[0]['metadata'].get('injury_type'):
                        return guide
        
        return None

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import faiss
import torch
import numpy as np
from sklearn.metrics import ndcg_score

class RetrievalPipeline:
    def __init__(self, embedding_model, ranking_model, corpus, k=10):
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.ranking_model = AutoModelForSequenceClassification.from_pretrained(ranking_model) if ranking_model else None
        self.ranking_tokenizer = AutoTokenizer.from_pretrained(ranking_model) if ranking_model else None
        
        # Normalize embeddings for inner product search
        self.corpus_embeddings = self.embedding_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        self.corpus_embeddings = torch.nn.functional.normalize(self.corpus_embeddings, p=2, dim=1)
        
        self.corpus = corpus
        self.k = k
        self.index = self.build_faiss_index()

    def build_faiss_index(self):
        # Use inner product instead of L2 distance
        dimension = self.corpus_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(self.corpus_embeddings.cpu().numpy().astype('float32'))
        return index

    def retrieve(self, query):
        # Get normalized query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        # Search FAISS index for top-k candidates
        D, I = self.index.search(query_embedding.cpu().numpy().astype('float32'), self.k)
        candidates = [self.corpus[i] for i in I[0]]
        
        # If ranking model is available, rank the retrieved documents
        if self.ranking_model:
            inputs = self.ranking_tokenizer(
                [query] * len(candidates),
                candidates,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            with torch.no_grad():
                scores = self.ranking_model(**inputs).logits.squeeze(-1)
            ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        else:
            ranked_results = list(zip(candidates, D[0]))  # Use distance as the score
        
        return ranked_results

def evaluate_pipeline(pipeline, queries, relevant_docs, num_queries=10):
    total_relevant = 0
    total_retrieved_relevant = 0
    
    for i, (query, rel_docs) in enumerate(zip(queries[:num_queries], relevant_docs[:num_queries])):
        print(f"\nQuery {i+1}: {query}")
        results = pipeline.retrieve(query)
        retrieved_docs = [doc for doc, _ in results]
        
        print("Relevant documents:")
        for j, answers in enumerate(rel_docs['text']):
            print(f"  {j+1}. {answers}")
        
        print("\nTop 3 retrieved documents:")
        for j, (doc, score) in enumerate(results[:3]):
            print(f"  {j+1}. Score: {score:.4f}, Document: {doc[:100]}...")
        
        query_relevant = set(rel_docs['text'])
        retrieved_relevant = set(retrieved_docs) & query_relevant
        
        total_relevant += len(query_relevant)
        total_retrieved_relevant += len(retrieved_relevant)
        
        print(f"\nRelevant documents retrieved: {len(retrieved_relevant)} out of {len(query_relevant)}")
    
    if total_relevant > 0:
        recall = total_retrieved_relevant / total_relevant
        print(f"\nOverall Recall@10: {recall:.4f}")
    else:
        print("\nNo relevant documents in the evaluation set.")

# Load and prepare dataset
print("Loading dataset...")
dataset = load_dataset("squad", split="validation")
corpus = dataset["context"]
queries = dataset["question"]
relevant_docs = dataset["answers"]

print(f"Corpus size: {len(corpus)}")
print(f"Number of queries: {len(queries)}")

# Define models to evaluate
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
ranking_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Evaluation
print(f"\nEvaluating: Embedding={embedding_model}, Ranking={ranking_model}")
pipeline = RetrievalPipeline(embedding_model, ranking_model, corpus)
evaluate_pipeline(pipeline, queries, relevant_docs)

print("\nEvaluating without ranking")
pipeline_no_ranking = RetrievalPipeline(embedding_model, None, corpus)
evaluate_pipeline(pipeline_no_ranking, queries, relevant_docs)

print("\nScript completed successfully")

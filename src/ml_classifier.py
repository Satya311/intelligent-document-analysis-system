import re
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class ConventionalMLAnalyzer:
    def __init__(self, num_clusters: int = 4):
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

    def perform_topic_clustering(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(chunks) < 2:
            for chunk in chunks:
                chunk["topic_cluster"] = 0
                chunk["topic_name"] = "General Topic"
            return {"cluster_titles": {0: "General Topic"}, "chunks": chunks}

        n_clusters = min(self.num_clusters, len(chunks))
        texts = [c["text"] for c in chunks]

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        cluster_titles = {}

        for i in range(n_clusters):
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-3:][::-1]
            top_words = feature_names[top_indices]
            cluster_titles[i] = f"Topic {i+1}: " + ", ".join(top_words).title()

        for idx, chunk in enumerate(chunks):
            chunk["topic_cluster"] = int(cluster_labels[idx])
            chunk["topic_name"] = cluster_titles[int(cluster_labels[idx])]

        return {"cluster_titles": cluster_titles, "chunks": chunks}

    def classify_content_type(self, text: str) -> str:
        lower_text = text.lower()
        if re.search(r'\b(defined as|refers to|means|is a|is defined|known as)\b', lower_text):
            return "Definition"
        elif re.search(r'\b(for example|for instance|such as|e\.g\.|case study|illustration)\b', lower_text):
            return "Example / Application"
        elif re.search(r'\b(in summary|in conclusion|to summarize|key takeaways|overall|finally)\b', lower_text):
            return "Summary / Key Takeaway"
        else:
            return "Explanation / Concept"

    def enrich_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clustering_results = self.perform_topic_clustering(chunks)
        enriched_chunks = clustering_results["chunks"]
        for chunk in enriched_chunks:
            chunk["content_type"] = self.classify_content_type(chunk["text"])
        return enriched_chunks

class DocumentRetriever:
    def __init__(self, vectorstore, top_k=3):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query):
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)

        chunks = []
        scores = []
        for doc, score in results:
            chunks.append(doc.page_content)
            scores.append(score)

        print("Retrieved Chunks:", len(chunks))
        return chunks, scores
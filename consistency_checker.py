import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

class ConsistencyChecker:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)
        self.stores = {}
        self.classifier = None

    def load_embeddings(self, book_name: str, filepath: str):
        """Load embeddings for a book."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.stores[book_name] = {
            'chunks': data['chunks'],
            'embeddings': data['embeddings'],
            'metadata': data['metadata']
        }

    def get_relevant_chunks(self, book_name: str, query: str, top_k: int = 3):
        """Get most relevant chunks for a query."""
        if book_name not in self.stores:
            return []

        store = self.stores[book_name]
        query_emb = self.model.encode(query)
        similarities = cosine_similarity([query_emb], store['embeddings'])[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'chunk': store['chunks'][idx],
                'metadata': store['metadata'][idx],
                'similarity': similarities[idx]
            })
        return results

    def extract_character_facts(self, char, chunks):
        """Extract facts about a specific character from chunks."""
        facts = []
        for chunk_data in chunks:
            text = chunk_data['chunk']

            # Simple fact extraction: sentences about the character
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                if re.search(re.escape(char), sent, re.IGNORECASE):
                    facts.append(sent.strip())

        return facts

    def check_consistency(self, book_name: str, char: str, statement: str):
        """Check if statement is consistent with book narrative."""
        # Get relevant chunks
        relevant = self.get_relevant_chunks(book_name, statement, top_k=5)

        if not relevant:
            return "unknown"

        # Extract facts about the character
        facts = self.extract_character_facts(char, relevant)

        if not facts:
            # No facts about the character found, check general similarity
            avg_sim = np.mean([chunk['similarity'] for chunk in relevant])
            return "consistent" if avg_sim > 0.3 else "contradict"

        # Simple consistency check: if statement contains keywords from facts, likely consistent
        statement_lower = statement.lower()
        consistent_score = 0
        total_score = 0

        for fact in facts:
            fact_lower = fact.lower()
            # Check for overlapping keywords
            stmt_words = set(re.findall(r'\b\w+\b', statement_lower))
            fact_words = set(re.findall(r'\b\w+\b', fact_lower))

            overlap = len(stmt_words.intersection(fact_words))
            total_score += len(stmt_words)

            if overlap > len(stmt_words) * 0.3:  # 30% overlap
                consistent_score += overlap

        if total_score == 0:
            return "unknown"

        consistency_ratio = consistent_score / total_score

        # Use classifier if trained
        if self.classifier:
            features = self.extract_features(book_name, char, statement, relevant)
            pred = self.classifier.predict([features])[0]
            return "consistent" if pred == 1 else "contradict"
        else:
            # Simple threshold
            return "consistent" if consistency_ratio > 0.2 else "contradict"

    def extract_features(self, book_name: str, char: str, statement: str, relevant_chunks):
        """Extract features for classifier."""
        features = []

        # Average similarity to relevant chunks
        similarities = [chunk['similarity'] for chunk in relevant_chunks]
        features.append(np.mean(similarities))
        features.append(np.max(similarities))

        # Length of statement
        features.append(len(statement.split()))

        # Number of relevant chunks
        features.append(len(relevant_chunks))

        # Character mention count in relevant chunks
        char_mentions = sum(1 for chunk in relevant_chunks if re.search(re.escape(char), chunk['chunk'], re.IGNORECASE))
        features.append(char_mentions)

        return features

    def train_classifier(self, train_df):
        """Train a classifier using the training data."""
        X = []
        y = []

        for _, row in train_df.iterrows():
            book = row['book_name']
            char = row['char']
            statement = row['content']
            label = 1 if row['label'] == 'consistent' else 0

            relevant = self.get_relevant_chunks(book, statement, top_k=5)
            if relevant:
                features = self.extract_features(book, char, statement, relevant)
                X.append(features)
                y.append(label)

        if X:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.classifier = LogisticRegression(random_state=42)
            self.classifier.fit(X_train, y_train)

            # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Classifier accuracy: {accuracy:.2f}")

    def predict_test(self, test_df, output_file: str):
        """Predict on test data and save results."""
        results = []

        for _, row in test_df.iterrows():
            book = row['book_name']
            char = row['char']
            statement = row['content']
            pred = self.check_consistency(book, char, statement)
            results.append({
                'id': row['id'],
                'label': pred
            })

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

def main():
    # Initialize checker
    checker = ConsistencyChecker()

    # Load embeddings
    checker.load_embeddings('In Search of the Castaways', './In_Search_of_the_Castaways_embeddings.pkl')
    checker.load_embeddings('The Count of Monte Cristo', './The_Count_of_Monte_Cristo_embeddings.pkl')

    # Load training data
    train_df = pd.read_csv('./Dataset/train.csv')

    # Train classifier
    checker.train_classifier(train_df)

    # Load test data
    test_df = pd.read_csv('./Dataset/test.csv')

    # Predict and save
    checker.predict_test(test_df, './test_predictions.csv')

if __name__ == "__main__":
    main()    # Load test data

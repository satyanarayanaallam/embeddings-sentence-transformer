# embeddings-sentence-transformer
# 📊 Sentence Embeddings with Sentence-BERT

This project demonstrates how to generate **sentence embeddings** using [Sentence-BERT](https://www.sbert.net/) and visualize them with **PCA** and **t-SNE**.  
It shows how semantically similar sentences cluster together in vector space.

---

## 🚀 Features
- Encode sentences into dense vectors using `sentence-transformers`.
- Compare sentence similarity with **cosine similarity**.
- Visualize embeddings in 2D using **PCA** + **t-SNE**.
- Example sentences include animals, finance, fruits, and polysemy cases (e.g., *bank*).

---

## 📦 Installation
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -U sentence-transformers scikit-learn matplotlib
```
```bash
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load SBERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example sentences
sentences = [
    "The cat sits on the mat.",
    "A dog is playing in the park.",
    "The stock market crashed yesterday.",
    "Investors are worried about inflation.",
    "Apples and bananas are types of fruit.",
    "The river bank was flooded after the storm.",
    "He deposited money in the bank."
]

# Encode sentences
embeddings = model.encode(sentences)

# Reduce dimensions with PCA
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(embeddings)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings_pca)

# Plot
plt.figure(figsize=(8,6))
for i, label in enumerate(sentences):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x+0.02, y+0.02, label, fontsize=9)

plt.title("Sentence Embeddings Visualized with t-SNE")
plt.show()
```
---
## 🔎 Expected Output
- Sentences about animals cluster together.
- Sentences about finance cluster together.
- The word bank shows polysemy: “river bank” vs “money bank” land in different regions.
- Fruits are far from finance/animals.
---
## 📂 Project Structure

- ├── README.md        # Project documentation
- ├── embeddings_sentence_transformer.ipynb # Main script for embeddings + visualization

---
## 📚 References
- Sentence-BERT Paper
- Sentence-Transformers Library
- Scikit-learn PCA
- t-SNE
---
## 🎯 Next Steps
- Extend to larger corpora (e.g., FAQs, customer feedback).
- Store embeddings in FAISS/ChromaDB for semantic search.
- Try domain-specific models (finance, biomedical, legal).
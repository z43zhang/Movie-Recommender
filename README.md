# 🎬 Movie Recommender System

A content-based movie recommender system built with **TF-IDF**, **Sentence Embeddings**, and a **Hybrid Search** engine. 

This project demonstrates how NLP can enhance recommendations — from fuzzy title matching to full semantic query search.

---

# 🚀 Features

- 🔍 **Fuzzy Title Search** — Corrects misspelled input titles using `fuzzywuzzy`
- 🧠 **TF-IDF + SVD Recommender** — Based on genres + keywords
- 🤖 **Sentence Embedding Recommender** — Uses `SentenceTransformer` (`MiniLM`) on genres + keywords + overview
- 🔀 **Hybrid Recommender** — Combines keyword and semantic signals with tunable weights
- 💬 **Natural Language Query** — Understands full-sentence queries
- 📊 **Consistent Scoring Output** — See and compare results from all 3 engines

---

# 📂 Dataset

- [TMDB 5000 Movie Dataset (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- File used: `tmdb_5000_movies.csv`

---

# 🛠️ Tech Stack

| Component               | Description                              |
|------------------------|------------------------------------------|
| `pandas`, `json`       | Data manipulation                        |
| `scikit-learn`         | TF-IDF, TruncatedSVD, Cosine Similarity  |
| `sentence-transformers`| Embeddings from MiniLM                   |
| `fuzzywuzzy`           | Fuzzy title matching                     |

---

# 🧪 How It Works

## 1. Preprocessing
- Genres and keywords are parsed from JSON strings
- Combined into searchable strings for TF-IDF and semantic input

## 2. Vector Representations
- `TF-IDF + SVD`: Reduces high-dimensional keyword space
- `Sentence Embeddings`: Captures meaning of text semantically

## 3. Recommendation

```python
recommend_tfidf()        # Based on keyword similarity
recommend_embedding()    # Based on semantic meaning
hybrid_search()          # Best of both
```
## 4. Poster + TMDB Links
- Fetch live movie posters via TMDB API
- Link users directly to TMDB movie pages

---

# 🧪 Example Query

## 🎬 Input:  
`"I want to watch Marvel movies"`

## 📘 TF-IDF Recommendations:
```text
The Book of Eli                                    | Score: 0.9305
World War Z                                        | Score: 0.9302
Battle for the Planet of the Apes                  | Score: 0.9015
Terminator 2: Judgment Day                         | Score: 0.9005
Mad Max                                            | Score: 0.8904
Resident Evil: Extinction                          | Score: 0.8810
Mad Max: Fury Road                                 | Score: 0.8768
eXistenZ                                           | Score: 0.8728
Rise of the Planet of the Apes                     | Score: 0.8703
Def-Con 4                                          | Score: 0.8688
```
### 📘 Results Analysis: High scores, but off-topic

✅ TF-IDF only sees **exact token overlap**.

✅ "Marvel" likely doesn't appear often in the dataset, or it’s not part of the `genres` or `keywords`.

✅ So it defaults to high-frequency terms like *"action"*, *"fight"*, *"apocalypse"*, etc.

❌ It **misses** actual Marvel movies like *Avengers* or *Iron Man*.

## 🤖 Embedding Recommendations:
```text
28 Weeks Later                                     | Score: 0.5742
28 Days Later                                      | Score: 0.5582
Contagion                                          | Score: 0.5519
World War Z                                        | Score: 0.5143
Doomsday                                           | Score: 0.4868
Resident Evil: Retribution                         | Score: 0.4851
Resident Evil                                      | Score: 0.4786
Dawn of the Planet of the Apes                     | Score: 0.4760
Resident Evil: Apocalypse                          | Score: 0.4724
Self/less                                          | Score: 0.4552
```
### 🤖 Results Analysis: Better intent, but not precise

✅ Sentence embeddings understand the query has *"action"*, *"franchise"*, *"virus"* themes.

✅ It only captures the **tone** of Marvel (action-packed, global threats)

❌ Doesn’t **anchor on the Marvel universe** (characters, studios, etc.).

❌ It’s **too general** — confusing Marvel with zombie/apocalypse movies.

## 🔀 Hybrid Recommendations (alpha = 0.8):
```text
Captain America: Civil War                         | Score: 0.3465
Avengers: Age of Ultron                            | Score: 0.3345
The Avengers                                       | Score: 0.3200
X2                                                 | Score: 0.3157
X-Men: Days of Future Past                         | Score: 0.3056
X-Men                                              | Score: 0.3054
Captain America: The Winter Soldier                | Score: 0.3014
Special                                            | Score: 0.2966
X-Men Origins: Wolverine                           | Score: 0.2962
Thor: The Dark World                               | Score: 0.2905
```
### 🔀 Results Analysis: On-point, accurate

✅ Blends:
  - **TF-IDF** for recognizing “Marvel” in keywords, genres, or title
  - **Embeddings** for understanding the *intent* (“superhero movies, cinematic universe, action teams”)

🎯 It’s the **only method** that actually understands *you want Marvel-brand superhero movies*, not just action/zombies.

---

# 📊 Method Comparison

| Method     | Pros                                       | Cons                                      |
|------------|--------------------------------------------|-------------------------------------------|
| **TF-IDF** | Fast, good for keyword matching            | Doesn't understand intent/context         |
| **Embedding** | Captures sentence meaning & tone       | Lacks grounding in actual movie metadata  |
| **Hybrid** | 🎯 Balances meaning + metadata = accurate  | Slightly slower, needs alpha tuning       |

> 💡 The Hybrid method prioritizes Marvel movies based on both semantic intent and metadata — unlike TF-IDF or Embedding alone.

---

# 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/z43zhang/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt

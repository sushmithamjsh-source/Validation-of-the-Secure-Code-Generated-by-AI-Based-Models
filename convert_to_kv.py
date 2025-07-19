from gensim.models import Word2Vec

# Load your existing model (even if erroring here, run this in original environment where it works)
model = Word2Vec.load("models/word2vec_withString10-200-300.model")

# Save just the word vectors (no training state)
model.wv.save("models/word2vec_vectors.kv")

print("✅ Saved word vectors to models/word2vec_vectors.kv")



# w2v_model = Word2Vec.load("models/word2vec_withString10-200-300.model")
# tries to load a full Word2Vec model, including training internals. That includes a broken reference to __randomstate_ctor, which causes the error.
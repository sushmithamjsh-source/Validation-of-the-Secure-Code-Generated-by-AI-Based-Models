import gensim
from gensim.models import KeyedVectors

# Path to your .model file
MODEL_PATH = "models/word2vec_withString10-200-300.model"
KV_OUTPUT = "models/word2vec_vectors.kv"

# 🔄 Load model using gensim's internal unpickler
model = gensim.utils.unpickle(MODEL_PATH)

# ✅ Save just the word vectors
model.wv.save(KV_OUTPUT)

print("✅ Saved word vectors to:", KV_OUTPUT)

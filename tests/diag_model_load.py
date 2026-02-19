from sentence_transformers import SentenceTransformer
import time

print("Checking model load...")
start = time.time()
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
end = time.time()
print(f"Model loaded in {end-start:.2f}s")

vec = model.encode(["Apple"], convert_to_numpy=True)
print(f"Successfully encoded: {vec.shape}")

import pickle
import os

meta_path = 'company_matcher_cache/e1894e93a84bbc84a9ec980508a5fec4_loc_metadata.pkl'
if not os.path.exists(meta_path):
    print(f"Error: {meta_path} not found")
    exit(1)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

print(f"Old Version: {meta.get('cache_version')}")
meta['cache_version'] = 'v4.1_location_decoupled'
print(f"New Version: {meta['cache_version']}")

with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print("Metadata updated successfully.")

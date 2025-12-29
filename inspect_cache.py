import pickle
import os

cache_dir = 'company_matcher_cache'
if not os.path.exists(cache_dir):
    print(f"Error: {cache_dir} not found")
    exit(1)

files = [f for f in os.listdir(cache_dir) if f.endswith('_metadata.pkl')]
for f in files:
    path = os.path.join(cache_dir, f)
    try:
        with open(path, 'rb') as f_in:
            meta = pickle.load(f_in)
            print(f"File: {f}")
            print(f"  Model: {meta.get('model_name')}")
            print(f"  Version: {meta.get('cache_version')}")
            print(f"  Companies: {meta.get('num_companies', 0):,}")
            print(f"  Location Data: {meta.get('has_location_data', False)}")
            print("-" * 40)
    except Exception as e:
        print(f"Error reading {f}: {e}")

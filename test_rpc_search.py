#!/usr/bin/env python3
"""Test RPC search."""
import sys
sys.path.insert(0, 'src')
from finetuner.core.cache_rpc import connect

server = connect()
result = server.search('IBM', top_k=3)

print('Search results:')
for m in result['results']:
    print(f"  {m['name']}: {m['score']:.3f}")

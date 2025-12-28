---
description: Clear the company matcher cache
---

// turbo-all
1. To clear all cached embeddings and indices:
   ```powershell
   Remove-Item -Recurse -Force company_matcher_cache\*
   ```
2. Note: The next run of the matcher will take ~45-60 minutes to rebuild the index for 2.9M companies.
3. You can also clear cache for a specific file by calling `matcher.clear_cache(cache_key)`.

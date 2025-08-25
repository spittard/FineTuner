# Company Name Matcher Web Application

A modern, responsive web application that provides an intuitive interface for company name matching using advanced AI-powered similarity search.

## Features

- 🎯 **Smart Company Matching**: Uses sentence transformers for semantic understanding
- 🚀 **Fast Performance**: Leverages FAISS for efficient similarity search
- 💾 **Intelligent Caching**: Persists embeddings and indexes to avoid recalculation
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations
- 📊 **Detailed Results**: Shows rank, company name, likeness percentage, and match rationale

## Screenshots

The web application displays results in a format similar to this:

| Rank | Company Name | Likeness % | Match Rationale |
|------|--------------|------------|-----------------|
| 1 | Essential Healthcare Solutions | 100% | Direct name match with domain-specific suffix |
| 2 | Essential Pharmaceuticals LLC | 94% | Same root + healthcare/pharma domain |
| 3 | Essential Business Solutions | 91% | Shared branding + consulting/service context |

## Prerequisites

- Python 3.8 or higher
- `training_data.json` file with company names (from your existing dataset)
- Internet connection (for downloading sentence transformer models on first run)

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure your dataset is available**:
   - Make sure `training_data.json` exists in the project directory
   - The file should contain company names in the format: `[{"Company Name": "Company Name"}, ...]`

## Running the Application

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Open your web browser**:
   - Navigate to: `http://localhost:5000`
   - The application will automatically load your company data and build the search index

3. **Start searching**:
   - Enter a company name in the search box
   - Select the number of results you want (5, 10, 15, or 20)
   - Click "Search" or press Enter

## How It Works

### Backend (Flask)
- **Data Loading**: Automatically loads company names from `training_data.json`
- **Index Building**: Creates semantic embeddings using sentence transformers
- **Caching**: Persists embeddings and FAISS index for fast subsequent searches
- **Search API**: Provides RESTful endpoint for company matching

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Bootstrap-based layout that works on all devices
- **Real-time Status**: Shows system readiness and company count
- **Interactive Results**: Displays results in a professional table format
- **Error Handling**: Graceful error messages and loading states

### AI Matching Engine
- **Semantic Understanding**: Uses `all-MiniLM-L6-v2` model for context-aware matching
- **Similarity Scoring**: FAISS provides fast cosine similarity calculations
- **Smart Rationale**: Generates human-readable explanations for matches

## API Endpoints

- **`GET /`**: Main application page
- **`POST /search`**: Company search endpoint
  - Parameters: `query` (company name), `top_k` (number of results)
- **`GET /status`**: System status and company count

## Configuration

### Customizing the Model
You can change the sentence transformer model by modifying the `CompanyMatcher` initialization in `app.py`:

```python
matcher = CompanyMatcher(model_name='all-MiniLM-L6-v2')
```

### Adjusting Cache Settings
The cache directory and settings are configurable in `CompanyMatcher.py`:

```python
self.cache_dir = "company_matcher_cache"  # Change cache location
```

## Performance

- **First Run**: ~30 seconds (builds index and caches data)
- **Subsequent Searches**: ~1-2 seconds (uses cached index)
- **Memory Usage**: ~100-200MB for 1000 companies
- **Scalability**: Handles thousands of companies efficiently

## Troubleshooting

### Common Issues

1. **"System not ready" error**:
   - Ensure `training_data.json` exists and is valid JSON
   - Check that the file contains company names in the correct format

2. **Slow first search**:
   - This is normal - the system needs to build the index once
   - Subsequent searches will be much faster

3. **Memory errors**:
   - Reduce the number of companies in your dataset
   - Use a smaller sentence transformer model

### Debug Mode
Run with debug enabled for detailed error messages:
```bash
python app.py
```

## Development

### Project Structure
```
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main web interface
├── CompanyMatcher.py     # AI matching engine
├── requirements.txt      # Python dependencies
├── training_data.json    # Company dataset
└── company_matcher_cache/ # Cached embeddings and indexes
```

### Adding Features
- **New Search Options**: Modify the search form in `index.html`
- **Additional Results**: Extend the results table structure
- **Custom Styling**: Update CSS in the `<style>` section
- **API Enhancements**: Add new routes in `app.py`

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Security Notes

- The application runs on `localhost` by default
- No authentication is implemented - suitable for internal/development use
- For production deployment, consider adding:
  - HTTPS
  - User authentication
  - Rate limiting
  - Input validation

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify your dataset format matches the expected structure
3. Ensure all dependencies are properly installed
4. Check that the cache directory has write permissions

---

**Enjoy using your AI-powered Company Name Matcher! 🚀**

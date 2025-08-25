# Auto-Reload Features for Company Name Matcher

The web application now includes automatic resource reloading capabilities to ensure your data stays up-to-date without manual intervention.

## Features

### 🔄 Automatic Data Detection
- **File Change Monitoring**: Automatically detects when `training_data.json` is modified
- **Smart Caching**: Only reloads data when necessary to maintain performance
- **Background Checks**: Performs periodic status checks every 10 seconds

### 🎯 Manual Reload
- **Reload Button**: Click the "Reload Data" button to force a manual refresh
- **Visual Feedback**: Shows loading state and success/error messages
- **Real-time Updates**: Immediately reflects changes in the interface

### 📊 Status Monitoring
- **Live Status**: Shows current system status and company count
- **Last Updated**: Displays when data was last refreshed
- **Auto-refresh**: Status updates automatically every 10 seconds

## How It Works

### Backend Auto-Reload
1. **File Modification Detection**: Monitors `training_data.json` modification time
2. **Intelligent Caching**: Stores file modification timestamps to detect changes
3. **Lazy Loading**: Only rebuilds the search index when data actually changes
4. **Performance Optimization**: Avoids unnecessary reloads during active searches

### Frontend Auto-Update
1. **Periodic Status Checks**: Automatically checks system status every 10 seconds
2. **Change Detection**: Updates interface when new data is detected
3. **User Feedback**: Shows clear indicators of system state and data freshness

## Usage

### Automatic Reload
- Simply modify your `training_data.json` file
- The system will automatically detect changes within 5-10 seconds
- No manual intervention required

### Manual Reload
1. Click the "Reload Data" button in the status section
2. Wait for the reload to complete
3. View the success message and updated company count

### Monitoring Status
- Green indicator: System ready with current data
- Red indicator: System needs attention or data unavailable
- Last updated timestamp shows when data was last refreshed

## Configuration

### Check Intervals
- **Data Check**: Every 5 seconds (backend)
- **Status Check**: Every 10 seconds (frontend)
- **File Monitoring**: Real-time modification detection

### Cache Settings
- **Embedding Cache**: Persisted to disk for fast reloads
- **Index Cache**: FAISS index cached for performance
- **File Timestamps**: Stored for change detection

## Benefits

1. **Always Current**: Data stays synchronized with your source files
2. **Performance**: Smart caching prevents unnecessary reloads
3. **User Experience**: Automatic updates without manual refresh
4. **Reliability**: Fallback to manual reload if automatic fails
5. **Monitoring**: Clear visibility into system status and data freshness

## Troubleshooting

### Data Not Updating
1. Check if `training_data.json` was actually modified
2. Verify file permissions and accessibility
3. Use manual reload button as fallback
4. Check browser console for error messages

### Performance Issues
1. Reduce check intervals if needed
2. Monitor memory usage with large datasets
3. Clear cache if experiencing issues

### Manual Override
- Use the reload button for immediate updates
- Check system status for current state
- Monitor last updated timestamp for verification

---

**The auto-reload system ensures your company matching data is always current and your search results are based on the latest information! 🚀**

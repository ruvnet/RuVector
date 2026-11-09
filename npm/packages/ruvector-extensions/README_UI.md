# 🎨 RuVector Graph Explorer UI

Interactive web-based visualization for exploring vector embeddings as a force-directed graph.

## ✨ Features

- 🌐 **Interactive force-directed graph** with D3.js
- 🖱️ **Drag, zoom, and pan** controls
- 🔍 **Search and filter** nodes by metadata
- 🎯 **Similarity queries** - click to find similar nodes
- 📊 **Metadata panel** with detailed node information
- ⚡ **Real-time updates** via WebSocket
- 📸 **Export** as PNG or SVG
- 📱 **Responsive design** for mobile devices
- 🎨 **Color-coded** nodes by category
- 📈 **Live statistics** dashboard

## 🚀 Quick Start

### Installation

```bash
npm install ruvector-extensions express ws
```

### Basic Usage

```typescript
import { RuvectorCore } from 'ruvector-core';
import { startUIServer } from 'ruvector-extensions';

// Initialize database
const db = new RuvectorCore({ dimension: 384 });

// Add some vectors
await db.add('doc1', embedding1, { label: 'Document 1', category: 'research' });
await db.add('doc2', embedding2, { label: 'Document 2', category: 'code' });

// Start UI server on port 3000
const server = await startUIServer(db, 3000);

// Open browser at http://localhost:3000
```

> **Database contract:** `UIServer` / `startUIServer(db, port, host = '127.0.0.1')` accept any object and call `db.add(id, embedding, metadata)`, `db.get(id)`, `db.list()`, `db.query(vector, { topK })` and `db.getStats()`. The `RuvectorCore` class shown above is illustrative — it is not exported by `ruvector-core`, and the `VectorDB` in `ruvector` 0.2.x uses a different method set (`insert`, `search`, `get`, ...), so you need an adapter object exposing the methods above. The server binds to loopback by default because its API is unauthenticated and includes a write endpoint (`POST /api/nodes`).

### Run Example

```bash
npm run example:ui
```

Then navigate to `http://localhost:3000` in your browser.

## 📸 Screenshots

### Main Interface
- Force-directed graph with interactive nodes
- Sidebar with search, filters, and statistics
- Real-time connection status indicator

### Features Demo
1. **Search**: Type in search box to filter nodes
2. **Select**: Click any node to view metadata
3. **Similarity**: Click "Find Similar Nodes" or double-click
4. **Export**: Save visualization as PNG or SVG
5. **Mobile**: Fully responsive on all devices

## 🎮 Controls

### Mouse/Touch
- **Click node**: Select and show metadata
- **Double-click node**: Find similar nodes
- **Drag node**: Reposition in graph
- **Scroll/Pinch**: Zoom in/out
- **Drag background**: Pan view

### Buttons
- **Search**: Filter nodes by ID or metadata
- **Similarity slider**: Adjust threshold (0-1)
- **Find Similar**: Query similar nodes
- **Export PNG/SVG**: Save visualization
- **Reset View**: Return to default zoom
- **Zoom +/-**: Zoom controls
- **Fit View**: Auto-fit graph to window

## 🌐 API Reference

### REST Endpoints

```bash
# Get graph data
GET /api/graph?max=100

# Search nodes
GET /api/search?q=query

# Find similar nodes
GET /api/similarity/:nodeId?threshold=0.5&limit=10

# Get node details
GET /api/nodes/:nodeId

# Add new node
POST /api/nodes
{
  "id": "node-123",
  "embedding": [0.1, 0.2, ...],
  "metadata": { "label": "Example" }
}

# Database statistics
GET /api/stats

# Health check
GET /health
```

### WebSocket Events

**Client → Server:**
```javascript
// Subscribe to updates
{ "type": "subscribe" }

// Request graph
{ "type": "request_graph", "maxNodes": 100 }

// Query similarity
{
  "type": "similarity_query",
  "nodeId": "node-123",
  "threshold": 0.5,
  "limit": 10
}
```

**Server → Client:**
```javascript
// Graph data
{ "type": "graph_data", "payload": { "nodes": [...], "links": [...] }}

// Node added
{ "type": "node_added", "payload": { "id": "...", "metadata": {...} }}

// Similarity results
{ "type": "similarity_result", "payload": { "nodeId": "...", "similar": [...] }}
```

## 🎨 Customization

### Node Colors

Customize in `/src/ui/app.js`:

```javascript
getNodeColor(node) {
    const colors = {
        'research': '#667eea',
        'code': '#f093fb',
        'docs': '#4caf50',
        'test': '#ff9800'
    };
    return colors[node.metadata?.category] || '#667eea';
}
```

### Styling

Edit `/src/ui/styles.css`:

```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #f093fb;
}
```

### Force Layout

Adjust physics in `/src/ui/app.js`:

```javascript
this.simulation
    .force('link', d3.forceLink().distance(100))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('collision', d3.forceCollide().radius(30));
```

## 🔧 Advanced Configuration

### Custom Server

```typescript
import { UIServer } from 'ruvector-extensions';

const server = new UIServer(db, 3000);

// Custom middleware
server.app.use('/custom', customRouter);

await server.start();
```

### Real-time Updates

```typescript
// Notify clients of changes
server.notifyGraphUpdate();

// Broadcast custom event
server.broadcast({
    type: 'custom_event',
    payload: { data: 'value' }
});
```

## 📱 Mobile Support

The UI is fully optimized for mobile:
- ✅ Touch gestures (pinch to zoom)
- ✅ Responsive sidebar layout
- ✅ Simplified mobile controls
- ✅ Optimized performance

## 🚀 Performance

### Large Graphs (1000+ nodes)

- Limit visible nodes to 500
- Use clustering for better performance
- Reduce force simulation iterations
- Hide labels at low zoom levels

### Optimizations

```javascript
// Reduce node limit
const maxNodes = 500;

// Faster convergence
this.simulation.alpha(0.5).alphaDecay(0.05);

// Conditional labels
label.style('display', d => zoom.scale() > 1.5 ? 'block' : 'none');
```

## 🌐 Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | ✅ Full |
| Firefox | 88+ | ✅ Full |
| Safari | 14+ | ✅ Full |
| Edge | 90+ | ✅ Full |
| Mobile Safari | 14+ | ✅ Full |
| Chrome Mobile | 90+ | ✅ Full |

## 📚 Documentation

- [UI Guide](https://github.com/ruvnet/ruvector/blob/main/npm/packages/ruvector-extensions/docs/UI_GUIDE.md) - Complete documentation
- [Examples](https://github.com/ruvnet/ruvector/tree/main/npm/packages/ruvector-extensions/src/examples) - Usage examples

## 🐛 Troubleshooting

### Graph not loading
- Check console for errors
- Verify database has data: `GET /api/stats`
- Check WebSocket connection status

### Slow performance
- Reduce max nodes in sidebar
- Clear filters
- Check network tab for slow API calls

### WebSocket issues
- Check firewall settings
- Verify port is accessible
- Look for server errors

## 📄 File Structure

```
src/
├── ui/
│   ├── index.html      # Main UI file
│   ├── app.js          # Client-side JavaScript
│   └── styles.css      # Styling
├── ui-server.ts        # Express server
└── examples/
    └── ui-example.ts   # Usage example
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📜 License

MIT License - see [LICENSE](https://github.com/ruvnet/ruvector/blob/main/LICENSE) file

## 🙏 Acknowledgments

- [D3.js](https://d3js.org/) - Graph visualization
- [Express](https://expressjs.com/) - Web server
- [WebSocket](https://github.com/websockets/ws) - Real-time updates

## 📞 Support

- 📖 [Documentation](https://github.com/ruvnet/ruvector)
- 🐛 [Issues](https://github.com/ruvnet/ruvector/issues)
- 💬 [Discussions](https://github.com/ruvnet/ruvector/discussions)

---

Built with ❤️ by the [ruv.io](https://ruv.io) team

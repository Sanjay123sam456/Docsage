# 🧠 DOCsage - AI-Powered Document Intelligence

**DOCsage** is an intelligent PDF document analyzer that leverages AI to help you understand, analyze, and extract insights from your documents. Upload a PDF and get instant summaries, keyword extraction, smart suggestions, and interactive Q&A capabilities.

![DOCsage Banner](https://img.shields.io/badge/AI-Powered-22c55e?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi) ![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## ✨ Features

### 📄 Document Processing
- **PDF Upload**: Drag & drop or click to upload PDF documents
- **Text Extraction**: Automatic text extraction from uploaded PDFs
- **Document Preview**: View extracted text preview instantly

### 🤖 AI-Powered Analysis
- **Smart Summarization**: Generate concise bullet-point summaries
- **Keyword Extraction**: Identify key topics, skills, and entities
- **Intelligent Suggestions**: Get context-aware improvement recommendations
- **Interactive Q&A**: Ask questions about your document and get accurate answers

### 💾 User Experience
- **Save Analysis**: Bookmark important analysis results
- **Chat History**: Keep track of your Q&A conversations
- **Modern UI**: Beautiful, responsive dark-themed interface
- **Real-time Feedback**: Loading indicators and status updates

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/docsage.git
cd docsage
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the project root:
```env
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional
OPENROUTER_MODEL=openai/gpt-4o-mini
PORT=8000
MAX_CONTEXT_CHARS=6000
LOG_LEVEL=INFO
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
```
http://localhost:8000
```

---

## 📁 Project Structure

```
docsage/
│
├── app.py              # FastAPI backend server
├── index.html          # Frontend UI (HTML/CSS/JS)
├── requirements.txt    # Python dependencies
├── .env               # Environment configuration (create this)
├── .env.example       # Example environment file
├── uploads/           # Temporary upload directory (auto-created)
└── README.md          # This file
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **PyPDF2**: PDF text extraction
- **OpenRouter**: AI model integration (supports multiple LLM providers)
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Modern CSS**: Gradient backgrounds, glassmorphism effects
- **Responsive Design**: Mobile-friendly interface

---

## 📡 API Endpoints

### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

Response: {
  "message": "File uploaded successfully",
  "textPreview": "...",
  "totalLength": 12345
}
```

### Ask Question
```http
POST /api/ask
Content-Type: application/json

Body: {
  "question": "What is this document about?"
}

Response: {
  "answer": "This document discusses..."
}
```

### Get Summary
```http
GET /api/summary

Response: {
  "summary": "• Point 1\n• Point 2\n..."
}
```

### Extract Keywords
```http
GET /api/keywords

Response: {
  "raw": "keyword1, keyword2, ...",
  "keywords": ["keyword1", "keyword2", ...]
}
```

### Get Suggestions
```http
GET /api/suggest

Response: {
  "suggestions": "Document Type: Resume\n\n1. Suggestion..."
}
```

### Health Check
```http
GET /health

Response: {
  "ok": true,
  "openrouter_key_set": true,
  "model": "openai/gpt-4o-mini"
}
```

---

## 🔧 Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | **Required** |
| `OPENROUTER_MODEL` | AI model to use | `openai/gpt-4o-mini` |
| `OPENROUTER_URL` | OpenRouter API endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| `PORT` | Server port | `8000` |
| `MAX_CONTEXT_CHARS` | Max characters sent to AI | `6000` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## 🎯 Use Cases

- **Resume Analysis**: Extract skills, get improvement suggestions
- **Research Papers**: Summarize findings, extract key concepts
- **Legal Documents**: Q&A on complex terms and clauses
- **Business Reports**: Quick summaries and key metrics extraction
- **Study Notes**: Generate summaries and test yourself with questions

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 TODO / Future Enhancements

- [ ] Support for multiple document formats (DOCX, TXT, etc.)
- [ ] User authentication and multi-user support
- [ ] Document persistence and history
- [ ] Export analysis results (PDF, DOCX)
- [ ] Vector database integration for better context retrieval
- [ ] Batch document processing
- [ ] Custom AI model selection in UI
- [ ] Document comparison features
- [ ] Citation and reference extraction

---

## 🐛 Known Limitations

- Currently supports PDF files only
- Single document at a time (no multi-doc support)
- In-memory storage (document cleared on new upload)
- Context window limited to `MAX_CONTEXT_CHARS` (may truncate large documents)
- No authentication (not production-ready for multi-user scenarios)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [Sanjay123sam456](https://github.com/Sanjay123sam456)


---

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai/) for AI model access
- [FastAPI](https://fastapi.tiangolo.com/) for the amazing web framework
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF processing

---

## 📸 Screenshots

### Upload & Preview
![Upload Interface](screenshots/upload.png)

### Analysis Tools
![Analysis Tools](screenshots/analysis.png)

### Interactive Q&A
![Chat Interface](screenshots/chat.png)

---

## ⚡ Performance Tips

1. **Optimize Context**: Adjust `MAX_CONTEXT_CHARS` based on your model's context window
2. **Model Selection**: Use faster models (like GPT-4o-mini) for quick responses
3. **Chunking**: For large documents, consider implementing chunking strategies
4. **Caching**: Consider caching frequent queries for better performance

---

## 🔒 Security Notes

- Never commit your `.env` file with API keys
- Use environment variables for sensitive data
- Implement rate limiting for production deployments
- Add input validation and sanitization
- Consider using authentication for production use

---

## 💬 Support

If you encounter any issues or have questions:
- Open an [Issue](https://github.com/yourusername/docsage/issues)
- Check existing issues for solutions
- Read the [FAQ](#faq) section

---

## FAQ

**Q: Can I use models other than GPT-4o-mini?**  
A: Yes! Set `OPENROUTER_MODEL` in your `.env` to any model supported by OpenRouter.

**Q: Why am I getting "demo mode" responses?**  
A: Your `OPENROUTER_API_KEY` is not set correctly in the `.env` file.

**Q: Can I process multiple PDFs at once?**  
A: Currently no, but this is on our roadmap for future updates.

**Q: Is my document data stored anywhere?**  
A: No, documents are processed in-memory and cleared on new uploads.

---

<div align="center">

**Made with ❤️ and ☕**

⭐ Star this repo if you find it useful!

</div># Docsage

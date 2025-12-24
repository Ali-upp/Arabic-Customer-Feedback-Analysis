#!/bin/bash

# تشغيل خادم تحليل رضا العملاء
# Run Arabic Feedback Analyzer Server

cd "$(dirname "$0")" || exit 1

echo "🚀 بدء تشغيل الخادم..."
echo "   Starting server on http://127.0.0.1:8000"
echo ""

python3 -m uvicorn app.backend:app --host 127.0.0.1 --port 8000 --reload

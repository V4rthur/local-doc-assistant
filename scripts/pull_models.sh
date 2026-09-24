#!/usr/bin/env bash
set -e
echo "Pulling Ollama models..."
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
echo "Reranker (bge-reranker-v2-m3) downloads on first use via HuggingFace."
echo "Done."
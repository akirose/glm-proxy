#!/bin/zsh

# Run the program
uv run uvicorn server:app --host 127.0.0.1 --port 8082 --reload

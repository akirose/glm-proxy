# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based proxy server that provides OpenAI API compatibility for GLM models (GLM-4, GLM-4.5, etc.). The key feature is automatic parsing of GLM's XML-formatted tool calls into OpenAI's JSON format, enabling seamless tool/function calling integration.

**Core purpose**: Bridge the gap between GLM's native XML tool call format (`<tool_call>...</tool_call>`) and the OpenAI API's expected JSON format for tool calls.

## Development Commands

### Running the Server

```bash
# Local development with hot reload
uv run uvicorn server:app --host 127.0.0.1 --port 8082 --reload

# Or use the convenience script
./run.sh

# Docker build and run
docker build -t glm-proxy .
docker run -p 8082:8082 glm-proxy
```

### Dependencies

This project uses `uv` for dependency management. Install dependencies with:

```bash
pip install --upgrade uv
uv sync --locked
```

Main dependencies: FastAPI, uvicorn, httpx, pydantic, python-dotenv

## Environment Configuration

Required environment variables in `.env`:

- `OPENAI_API_KEY`: API key for the backend LLM service
- `OPENAI_BASE_URL`: Base URL for the backend API (defaults to `http://127.0.0.1:1234/v1`)
- `PORT`: Server port (optional, defaults to 8082)

Note: Despite the "OPENAI" naming, these can point to any OpenAI-compatible API endpoint.

## Architecture

### Core Components

1. **FastAPI Server** (`server.py`): Main application entry point with a single endpoint `/v1/chat/completions`

2. **GLMToolParser Class** (`server.py:139-263`): The heart of the proxy - parses GLM's XML tool call format
   - Regex-based extraction of `<tool_call>` blocks
   - Argument parsing from `<arg_key>/<arg_value>` pairs
   - Automatic type conversion (JSON, Python literals, strings)
   - Generates OpenAI-compatible tool call IDs and structure

3. **Request Flow**:
   - Client sends OpenAI-format request →
   - Proxy forwards to backend (GLM or OpenAI API) →
   - If GLM model and response contains `<tool_call>` tags →
   - Parser extracts and converts to OpenAI format →
   - Returns OpenAI-compatible response to client

### Model Detection

The function `is_glm_model()` at `server.py:335` checks if a model name contains "glm" (case-insensitive) to determine whether to apply GLM tool parsing.

**Examples of detected GLM models:**
- `glm-4`
- `glm-4-plus`
- `glm-4.5`
- `glm-4.5-air`
- Any model name containing "glm" substring

**Non-GLM models** (passed through without tool parsing):
- `gpt-4`, `gpt-3.5-turbo`
- Any other model name without "glm"

### Streaming vs Non-Streaming

- **Non-streaming** (`call_openai_api`, `server.py:339`): Parses complete response, extracts all tool calls at once
- **Streaming** (`handle_streaming_response`, `server.py:441`):
  - Accumulates all content chunks during streaming
  - Filters out `<tool_call>` XML tags from streamed content in real-time
  - Sends only clean text content to client as it arrives
  - When `[DONE]` is received, parses accumulated content for tool calls
  - Sends tool calls as final chunk before completion if any were found
  - This ensures clients never see raw XML tags, only clean text and final tool calls

### Content Normalization

The proxy handles content arrays (multi-modal format) by extracting text from `[{"type": "text", "text": "..."}]` structures and joining them into simple strings for backend compatibility (`server.py:363-374`).

## Key Implementation Details

### GLM Tool Call Format

GLM models (GLM-4, GLM-4.5) output:
```xml
<tool_call>function_name
<arg_key>param1</arg_key>
<arg_value>value1</arg_value>
<arg_key>param2</arg_key>
<arg_value>value2</arg_value>
</tool_call>
```

Converted to OpenAI format:
```json
{
  "tool_calls": [{
    "id": "call_abc123...",
    "type": "function",
    "function": {
      "name": "function_name",
      "arguments": "{\"param1\": \"value1\", \"param2\": \"value2\"}"
    }
  }]
}
```

### Argument Value Deserialization

The parser tries three approaches in order (`_deserialize_value`, `server.py:165`):
1. JSON parsing (for `{"key": "value"}` or `[1, 2, 3]`)
2. Python literal eval (for Python-style literals)
3. Return as string (fallback)

This enables handling of complex nested structures in GLM tool arguments.

### Streaming Implementation Details

The streaming handler (`handle_streaming_response`, `server.py:441-565`) implements intelligent content filtering:

1. **Accumulation**: All content deltas are accumulated in `accumulated_content`
2. **Real-time Filtering**: For each chunk:
   - Remove all complete `<tool_call>...</tool_call>` blocks using regex
   - Detect incomplete tool calls (started but not yet closed) and truncate before the opening tag
   - Track what content has been sent (`sent_length`) to avoid duplication
   - Send only new, clean content to the client
3. **Final Processing**: When `[DONE]` is received:
   - Parse accumulated content for tool calls
   - If tool calls found, send them as a final delta with `finish_reason: "tool_calls"`
   - Ensures the client receives a complete, properly formatted response

This approach provides a seamless experience where GLM's XML format is completely transparent to the client.

## Logging

- Default log level: WARN (can set to DEBUG for troubleshooting)
- Beautiful colored console output for each request via `log_request_beautifully()` (`server.py:64-97`)
  - Uses ANSI color codes defined in `Colors` class (`server.py:52-62`)
  - Shows method, endpoint, status, model, message count, tool count, and streaming status
- Request/response logging via middleware (at DEBUG level)
- Detailed GLM parser debug logs available when DEBUG enabled
- Validation error handler provides detailed error information (`server.py:119-132`)
- HTTP request logs are filtered out to reduce noise (`MessageFilter` class, `server.py:34-45`)

Example console output:
```
POST /v1/chat/completions ✓ 200 glm-4.5-air 3 msgs 2 tools stream
```

To enable detailed debugging, change log level at `server.py:22`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## API Compatibility

The server implements OpenAI's chat completion API:
- POST `/v1/chat/completions`: Main endpoint
- Supports all standard OpenAI parameters (temperature, top_p, max_tokens, tools, tool_choice, presence_penalty, frequency_penalty, etc.)
- Fully compatible with OpenAI client libraries
- Default timeout: 300 seconds (5 minutes) for backend API calls

## Error Handling

- Custom validation error handler (`server.py:119-132`) provides detailed error information
- HTTP errors from backend API are logged with full context (status, body, URL, payload)
- Stream errors are caught and formatted as SSE error messages
- All exceptions are properly logged with stack traces when DEBUG is enabled

## Docker Deployment

The Dockerfile:
- Uses Python latest base image
- Installs dependencies via `uv sync --locked`
- Exposes port 8082
- Runs with `--reload` flag for development

For production, remove `--reload` flag in the CMD directive.

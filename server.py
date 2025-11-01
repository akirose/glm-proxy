from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import uuid
import time
from dotenv import load_dotenv
import re
import ast
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to DEBUG for troubleshooting
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Disable verbose httpx logging

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "HTTP Request:",
        ]

        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

def log_request_beautifully(method: str, path: str, model: str, num_messages: int,
                            num_tools: int, status_code: int, stream: bool = False):
    """Log requests in a beautiful, concise format."""
    # Format the model name nicely
    model_display = f"{Colors.CYAN}{model}{Colors.RESET}"

    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}" if num_tools > 0 else ""
    messages_str = f"{Colors.BLUE}{num_messages} msgs{Colors.RESET}"
    stream_str = f"{Colors.YELLOW}stream{Colors.RESET}" if stream else ""

    # Format status code
    if status_code == 200:
        status_str = f"{Colors.GREEN}✓ {status_code}{Colors.RESET}"
    else:
        status_str = f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    # Build the log line with all components
    components = [f"{Colors.BOLD}{method}{Colors.RESET}", endpoint, status_str, model_display, messages_str]
    if tools_str:
        components.append(tools_str)
    if stream_str:
        components.append(stream_str)

    log_line = " ".join(components)

    # Print to console
    print(log_line)
    sys.stdout.flush()

app = FastAPI()

# Middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging"""
    logger.debug(f"Incoming request: {request.method} {request.url.path}")
    logger.debug(f"Headers: {dict(request.headers)}")

    # For POST requests, we'll log the body in the endpoint itself
    # to avoid consuming the request stream

    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

# Custom exception handler for validation errors
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed information"""
    logger.error(f"Validation error for {request.url.path}")
    logger.error(f"Errors: {exc.errors()}")
    logger.error(f"Body: {exc.body}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(exc.body)[:500] if exc.body else None
        }
    )

# Get API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

# GLM Tool Parser for handling GLM-4's tool call format
class GLMToolParser:
    """Parser for GLM-4 model's tool call format.
    
    GLM-4 uses XML-like tags for tool calls:
    <tool_call>function_name
    <arg_key>param1</arg_key>
    <arg_value>value1</arg_value>
    ...
    </tool_call>
    """
    
    def __init__(self):
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        
        # Regex patterns for parsing GLM tool calls
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )
        
        logger.debug("GLMToolParser initialized")
    
    def _deserialize_value(self, value: str) -> Any:
        """Try to deserialize a string value to appropriate Python type."""
        value = value.strip()
        
        # Try JSON parsing first
        try:
            return json.loads(value)
        except:
            pass
        
        # Try Python literal eval
        try:
            return ast.literal_eval(value)
        except:
            pass
        
        # Return as string if all else fails
        return value
    
    def parse_tool_calls(self, model_output: str) -> List[Dict[str, Any]]:
        """Extract tool calls from GLM model output.
        
        Args:
            model_output: Raw output from GLM model
            
        Returns:
            List of tool call dictionaries with id, function name, and arguments in OpenAI format
        """
        matched_tool_calls = self.func_call_regex.findall(model_output)
        tool_calls = []
        
        logger.debug(f"GLM Parser: Found {len(matched_tool_calls)} potential tool calls")
        
        for match in matched_tool_calls:
            try:
                tc_detail = self.func_detail_regex.search(match)
                if not tc_detail:
                    logger.warning(f"GLM Parser: Failed to parse tool call structure: {match[:100]}")
                    continue
                
                tc_name = tc_detail.group(1).strip()
                tc_args_text = tc_detail.group(2)
                
                # Parse argument key-value pairs
                pairs = self.func_arg_regex.findall(tc_args_text)
                arg_dict = {}
                
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = self._deserialize_value(value.strip())
                    arg_dict[arg_key] = arg_val
                    logger.debug(f"GLM Parser: Parsed arg {arg_key}={arg_val} (type: {type(arg_val).__name__})")
                
                # Generate OpenAI-style tool call ID
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                
                # Format in OpenAI tool call structure
                tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": json.dumps(arg_dict)  # OpenAI expects JSON string
                    }
                })
                
                logger.debug(f"GLM Parser: Extracted tool call '{tc_name}' with {len(arg_dict)} arguments")
                
            except Exception as e:
                logger.error(f"GLM Parser: Error parsing tool call: {e}", exc_info=True)
                continue
        
        return tool_calls
    
    def extract_content_and_tools(self, model_output: str) -> tuple[str, List[Dict[str, Any]]]:
        """Separate text content and tool calls from GLM model output.
        
        Args:
            model_output: Raw output from GLM model
            
        Returns:
            Tuple of (text_content, tool_calls_list)
        """
        start_idx = model_output.find(self.tool_call_start_token)
        
        if start_idx == -1:
            # No tool calls found
            logger.debug("GLM Parser: No tool calls detected in output")
            return model_output, []
        
        # Extract content before first tool call
        content = model_output[:start_idx].strip()
        
        # Extract all tool calls
        tool_calls = self.parse_tool_calls(model_output[start_idx:])
        
        logger.debug(f"GLM Parser: Extracted {len(content)} chars of content and {len(tool_calls)} tool calls")
        
        return content, tool_calls

# Create global GLM tool parser instance
glm_tool_parser = GLMToolParser()

# OpenAI API Models
class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string

class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall

class ContentPart(BaseModel):
    """Content part for multi-modal messages"""
    type: str  # "text", "image_url", etc.
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Content can be either a string or an array of content parts (for multi-modal)
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None  # for function/tool messages
    tool_calls: Optional[List[ToolCall]] = None  # for assistant messages with tool calls
    tool_call_id: Optional[str] = None  # for tool role messages

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def is_glm_model(model: str) -> bool:
    """Check if the model is a GLM model that needs special parsing."""
    return "glm" in model.lower()

async def call_openai_api(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Call OpenAI API (or compatible API) and handle GLM tool parsing if needed."""
    
    # Determine if this is a GLM model
    is_glm = is_glm_model(request.model)
    
    # Choose appropriate API endpoint and key
    base_url = OPENAI_BASE_URL
    api_key = OPENAI_API_KEY
    logger.debug(f"Using OpenAI API for model: {request.model}")

    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    # Prepare request payload - use model_dump for Pydantic v2
    try:
        payload = request.model_dump(exclude_none=True)
    except AttributeError:
        # Fallback for Pydantic v1
        payload = request.dict(exclude_none=True)
    
    # Convert content arrays to strings for backend compatibility
    # Some clients send content as [{"type": "text", "text": "..."}]
    # but most backends expect simple strings
    if "messages" in payload:
        for message in payload["messages"]:
            if isinstance(message.get("content"), list):
                # Extract text from content parts
                text_parts = []
                for part in message["content"]:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and "text" in part:
                            text_parts.append(part["text"])
                        # Could handle other types (image_url, etc.) here
                # Join all text parts into a single string
                message["content"] = "\n".join(text_parts) if text_parts else ""
    
    logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make API call
    if request.stream:
        # Handle streaming response
        return await handle_streaming_response(base_url, headers, payload, is_glm)
    else:
        # Handle non-streaming response
        async with httpx.AsyncClient(timeout=300.0) as client:
            url = f"{base_url}/chat/completions"
            logger.debug(f"Calling API: {url}")
            
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                
                logger.debug(f"API Response Status: {response.status_code}")
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"API error: {response.status_code}")
                    logger.error(f"Response body: {error_detail}")
                    logger.error(f"Request URL: {url}")
                    logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
                    raise HTTPException(status_code=response.status_code, detail=error_detail)
                
                response_data = response.json()
                logger.debug(f"API Response data: {json.dumps(response_data, indent=2)[:500]}")
            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error calling API: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
        
        # If it's a GLM model and has tool calls, parse them
        if is_glm and response_data.get("choices"):
            for choice in response_data["choices"]:
                message = choice.get("message", {})
                content = message.get("content", "")
                
                if content and glm_tool_parser.tool_call_start_token in content:
                    # Parse GLM tool calls
                    text_content, tool_calls = glm_tool_parser.extract_content_and_tools(content)
                    
                    # Update message with parsed content
                    message["content"] = text_content if text_content else None
                    
                    if tool_calls:
                        message["tool_calls"] = tool_calls
                        # Update finish reason to tool_calls if tools were found
                        choice["finish_reason"] = "tool_calls"

                    logger.debug(f"GLM Parser: Converted {len(tool_calls)} tool calls to OpenAI format")

        return response_data

async def handle_streaming_response(base_url: str,
                                   headers: Dict, payload: Dict, is_glm: bool) -> StreamingResponse:
    """Handle streaming responses with GLM tool call parsing."""

    async def stream_generator():
        accumulated_content = ""
        sent_length = 0  # Track how much content we've already sent to client
        done_sent = False

        try:
            # Create a new client inside the generator to avoid scope issues
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(f"API error: {response.status_code} - {error_text}")
                        yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                        yield f"data: [DONE]\n\n"
                        return

                    logger.debug("Starting to stream response from backend...")

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]

                            if data_str.strip() == "[DONE]":
                                logger.debug("Stream completed - received [DONE] from backend")
                                # If we accumulated content and it's a GLM model, check for tool calls
                                if is_glm and accumulated_content:
                                    if glm_tool_parser.tool_call_start_token in accumulated_content:
                                        # Parse tool calls at the end
                                        text_content, tool_calls = glm_tool_parser.extract_content_and_tools(accumulated_content)

                                        if tool_calls:
                                            # Send tool calls as a final chunk
                                            tool_chunk = {
                                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": payload["model"],
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"tool_calls": tool_calls},
                                                    "finish_reason": "tool_calls"
                                                }]
                                            }
                                            yield f"data: {json.dumps(tool_chunk)}\n\n"
                                            logger.debug(f"GLM Stream Parser: Sent {len(tool_calls)} tool calls to client")

                                yield f"data: [DONE]\n\n"
                                done_sent = True
                                return

                            try:
                                chunk_data = json.loads(data_str)

                                # Handle GLM models - filter out tool call XML content
                                if is_glm:
                                    choices = chunk_data.get("choices", [])

                                    # Process only the first choice (standard for streaming)
                                    if choices and len(choices) > 0:
                                        choice = choices[0]
                                        delta = choice.get("delta", {})

                                        if "content" in delta and delta["content"]:
                                            # Accumulate content from this chunk
                                            accumulated_content += delta["content"]

                                            # Remove all <tool_call>...</tool_call> blocks from accumulated content
                                            clean_content = accumulated_content
                                            # Use regex to remove all complete tool call blocks
                                            clean_content = glm_tool_parser.func_call_regex.sub("", clean_content)

                                            # Also handle incomplete tool calls (started but not yet closed)
                                            tool_start_idx = clean_content.find(glm_tool_parser.tool_call_start_token)
                                            if tool_start_idx != -1:
                                                # Incomplete tool call, remove everything from the start token onwards
                                                clean_content = clean_content[:tool_start_idx]

                                            # Determine new content to send (only what we haven't sent yet)
                                            content_to_send = clean_content[sent_length:]

                                            if content_to_send:
                                                # Update delta with only the new clean content
                                                chunk_data["choices"][0]["delta"] = {"content": content_to_send}
                                                sent_length = len(clean_content)
                                                yield f"data: {json.dumps(chunk_data)}\n\n"
                                                logger.debug(f"GLM Stream: Sent {len(content_to_send)} chars (total sent: {sent_length})")
                                            # If no new content, don't send anything
                                        else:
                                            # No content in delta, forward as-is (might be role or other metadata)
                                            yield f"data: {json.dumps(chunk_data)}\n\n"

                                    # Don't send original chunk for GLM models - already processed
                                    continue

                                # Forward the chunk as-is (for non-GLM models)
                                yield f"data: {json.dumps(chunk_data)}\n\n"

                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse chunk: {data_str}")
                                continue

                    # If we finished the loop without sending [DONE], send it now
                    if not done_sent:
                        logger.debug("Stream ended without [DONE] from backend, sending it now")
                        yield f"data: [DONE]\n\n"
                    
        except Exception as e:
            logger.error(f"Error in stream_generator: {e}", exc_info=True)
            if not done_sent:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield f"data: [DONE]\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with GLM tool call support."""
    try:
        # Log request beautifully
        log_request_beautifully(
            "POST",
            "/v1/chat/completions",
            request.model,
            len(request.messages),
            len(request.tools) if request.tools else 0,
            200,  # Assuming success at this point
            request.stream or False
        )

        # Log detailed info at debug level
        logger.debug(f"Parsed request - model: {request.model}, stream: {request.stream}, tools: {len(request.tools) if request.tools else 0}")
        logger.debug(f"Messages count: {len(request.messages)}")

        # Log message content types for debugging
        for i, msg in enumerate(request.messages):
            content_type = type(msg.content).__name__ if msg.content is not None else "None"
            logger.debug(f"Message {i}: role={msg.role}, content_type={content_type}")

        result = await call_openai_api(request)

        if isinstance(result, StreamingResponse):
            return result
        else:
            return JSONResponse(content=result)
            
    except HTTPException as he:
        logger.error(f"HTTPException: {he.status_code} - {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8082))
    print(f"\n{Colors.BOLD}{Colors.GREEN}🚀 GLM Proxy Server{Colors.RESET}")
    print(f"{Colors.CYAN}Port:{Colors.RESET} {port}")
    print(f"{Colors.CYAN}Backend:{Colors.RESET} {OPENAI_BASE_URL}")
    print(f"{Colors.CYAN}Log Level:{Colors.RESET} {logging.getLevelName(logger.level)}")
    print(f"{Colors.DIM}Set logging.basicConfig level to DEBUG for detailed logs{Colors.RESET}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


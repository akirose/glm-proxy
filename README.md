# GLM Tool Call Proxy

OpenAI API 호환 서버로, GLM-4.5 모델의 도구 호출을 자동으로 파싱하여 OpenAI 형식으로 변환합니다.

## 주요 기능

1. **OpenAI API 호환**: 표준 OpenAI API 형식을 완벽하게 지원
2. **GLM Tool Parser**: GLM-4.5 모델의 XML 형식 도구 호출을 OpenAI 형식으로 자동 변환
3. **스트리밍 지원**: 실시간 스트리밍 응답 지원
4. **다중 모델 지원**: OpenAI 및 GLM 모델 모두 지원

## GLM Tool Call 변환

GLM-4.5는 다음과 같은 XML 형식으로 도구를 호출합니다:

```xml
<tool_call>function_name
<arg_key>param1</arg_key>
<arg_value>value1</arg_value>
<arg_key>param2</arg_key>
<arg_value>value2</arg_value>
</tool_call>
```

이 서버는 자동으로 이를 OpenAI 형식으로 변환합니다:

```json
{
  "tool_calls": [
    {
      "id": "call_abc123...",
      "type": "function",
      "function": {
        "name": "function_name",
        "arguments": "{\"param1\": \"value1\", \"param2\": \"value2\"}"
      }
    }
  ]
}
```

## 설치 및 설정

### 1. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가합니다:

```bash
# 백엔드 LLM API 설정 (OpenAI 호환 API)
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=http://127.0.0.1:1234/v1

# 서버 포트 (선택사항, 기본값: 8082)
PORT=8082
```

**참고**: `OPENAI_BASE_URL`은 OpenAI API뿐만 아니라 로컬 LLM 서버 (예: LM Studio, Ollama 등) 또는 다른 OpenAI 호환 API를 가리킬 수 있습니다.

### 2. 의존성 설치

이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다:

```bash
# uv 설치 (아직 설치하지 않은 경우)
pip install --upgrade uv

# 의존성 설치
uv sync --locked
```

또는 pip으로 직접 설치:

```bash
pip install fastapi uvicorn httpx pydantic python-dotenv
```

### 3. 서버 실행

#### 개발 모드 (자동 재시작)

```bash
uv run uvicorn server:app --host 127.0.0.1 --port 8082 --reload
```

또는 편의 스크립트 사용:

```bash
./run.sh
```

#### 일반 모드

```bash
python server.py
```

#### Docker 사용

```bash
# 이미지 빌드
docker build -t glm-proxy .

# 컨테이너 실행
docker run -p 8082:8082 --env-file .env glm-proxy
```

서버는 `http://localhost:8082`에서 실행됩니다.

## API 사용 예제

### 기본 채팅 완성 (Non-streaming)

```python
import requests
import json

url = "http://localhost:8082/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "glm-4.5-air",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

### 도구 호출 예제

```python
import requests
import json

url = "http://localhost:8082/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 요청 데이터
data = {
    "model": "glm-4.5-air",  # GLM 모델 사용
    "messages": [
        {"role": "user", "content": "What's the weather like in Seoul?"}
    ],
    "tools": tools,
    "tool_choice": "auto",
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

# GLM의 XML 형식 도구 호출이 자동으로 OpenAI 형식으로 변환됨
print(json.dumps(result, indent=2))
```

### 스트리밍 응답

```python
import requests
import json

url = "http://localhost:8082/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "glm-4.5-air",
    "messages": [
        {"role": "user", "content": "Tell me a short story"}
    ],
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str.strip() == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if chunk['choices'][0]['delta'].get('content'):
                    print(chunk['choices'][0]['delta']['content'], end='', flush=True)
            except json.JSONDecodeError:
                pass
```

### 도구 호출 결과 전달

```python
# 1. 첫 번째 요청 - 도구 호출 받기
response1 = requests.post(url, headers=headers, json={
    "model": "glm-4.5-air",
    "messages": [
        {"role": "user", "content": "What's the weather in Seoul?"}
    ],
    "tools": tools
})

result1 = response1.json()
tool_calls = result1['choices'][0]['message']['tool_calls']

# 2. 도구 실행 (예시)
tool_results = []
for tool_call in tool_calls:
    function_name = tool_call['function']['name']
    arguments = json.loads(tool_call['function']['arguments'])
    
    # 실제 도구 실행
    if function_name == 'get_weather':
        result = {"temperature": 20, "condition": "sunny"}
        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call['id'],
            "content": json.dumps(result)
        })

# 3. 도구 결과와 함께 다시 요청
messages = [
    {"role": "user", "content": "What's the weather in Seoul?"},
    result1['choices'][0]['message'],
    *tool_results
]

response2 = requests.post(url, headers=headers, json={
    "model": "glm-4.5-air",
    "messages": messages,
    "tools": tools
})

print(response2.json())
```

## API 엔드포인트

### POST /v1/chat/completions

채팅 완성 생성 - OpenAI Chat Completions API와 완벽하게 호환됩니다.

**요청 본문:**
```json
{
  "model": "string",
  "messages": [{"role": "string", "content": "string"}],
  "temperature": 1.0,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "max_tokens": null,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "tools": [],
  "tool_choice": "auto"
}
```

**응답 (Non-streaming):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "response text",
      "tool_calls": []
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

**응답 (Streaming):**
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: [DONE]
```

## GLM Tool Parser 동작 방식

1. **요청 처리**: 클라이언트가 OpenAI 형식으로 도구를 정의하여 요청
2. **API 호출**: GLM 모델이 XML 형식으로 도구 호출 응답
3. **자동 파싱**: GLM Tool Parser가 XML을 파싱하여 OpenAI 형식으로 변환
4. **응답 반환**: 표준 OpenAI 형식의 응답을 클라이언트에 반환

### 파싱 지원 기능

- ✅ 복수 도구 호출 파싱
- ✅ 복잡한 JSON 인자 파싱
- ✅ Python 리터럴 파싱 (리스트, 딕셔너리 등)
- ✅ 중첩된 객체 구조 지원
- ✅ 스트리밍 모드에서의 도구 호출 처리
- ✅ 텍스트 컨텐츠와 도구 호출 분리

## 로깅

서버는 기본적으로 WARN 레벨 로깅을 사용하며, 각 요청에 대해 아름다운 색상 포맷으로 로그를 출력합니다:

```
POST /v1/chat/completions ✓ 200 glm-4 3 msgs 2 tools stream
```

### 로그 레벨 변경

문제 해결을 위해 더 자세한 로그가 필요한 경우, `server.py`의 다음 코드에서 로그 레벨을 변경하세요:

```python
logging.basicConfig(
    level=logging.DEBUG,  # WARN에서 DEBUG로 변경
    format='%(asctime)s - %(levelname)s - %(message)s',
)
```

## 문제 해결

### GLM 도구 호출이 파싱되지 않음

1. 모델 이름에 "glm"이 포함되어 있는지 확인 (예: "glm-4.5", "glm-4.5-air")
   - 모델 이름에 "glm"이 없으면 자동 파싱이 활성화되지 않습니다
2. 백엔드 API가 GLM 모델을 제공하고 있는지 확인
3. 로그 레벨을 DEBUG로 변경하여 파싱 과정 확인:
   ```python
   logging.basicConfig(level=logging.DEBUG, ...)
   ```
4. GLM 응답에 `<tool_call>` 태그가 포함되어 있는지 확인

### 백엔드 API 연결 실패

1. `OPENAI_BASE_URL`이 올바르게 설정되었는지 확인
   - 로컬 LLM 서버 (예: LM Studio): `http://127.0.0.1:1234/v1`
   - 기타 OpenAI 호환 API: 해당 서비스의 base URL
2. `OPENAI_API_KEY`가 올바르게 설정되었는지 확인
3. 백엔드 서버가 실행 중인지 확인 (로컬 서버의 경우)
4. 네트워크 연결 확인 (방화벽, 프록시 설정 등)

### 스트리밍이 작동하지 않음

1. 백엔드 API가 스트리밍을 지원하는지 확인
2. 클라이언트에서 `stream=True` 설정 확인
3. 타임아웃 설정 확인 (기본값: 300초)

## 라이선스

MIT License

## 기여

버그 리포트와 기능 제안은 GitHub Issues를 사용합니다.

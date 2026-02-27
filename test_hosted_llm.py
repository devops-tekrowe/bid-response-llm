# export VLLM_LOGGING_LEVEL=INFO
# vllm serve Qwen/Qwen2.5-1.5B-Instruct \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --max-num-seqs 8

# activate environment
# ls -d */ | grep -E "venv|.venv|env"

# move to downloads directory
# cd ~/myenv
# source bin/activate



# to pull from the git
# cd projects/llm-bid-response-ai-dev
# cd onedrive-integration
# git pull
# To activate docker 
# docker compose up -d

# ssh command to connect to the server
# ssh -i bid-response-ai-dev_Hyperstack.pem ubuntu@149.36.1.74


from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://149.36.1.74:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response.choices[0].message.content)
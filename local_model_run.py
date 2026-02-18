from openai import OpenAI
BASE_URL = "http://localhost:12434/engines/v1"


client  = OpenAI(base_url=BASE_URL,api_key="anything")

Model_Name = "ai/llama3.2:1B-F16"
Prompt = "Explain the concept of AI in 500 words"


messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of the United States of America?"}
]

print("Calling the model...")
response = client.chat.completions.create(model=Model_Name,messages=messages)

print(response.choices[0].message.content)
print("Model response received")
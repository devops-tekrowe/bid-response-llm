from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12434/v1",
    api_key="test"
)

models = client.models.list()
print(models)

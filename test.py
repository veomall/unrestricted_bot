import g4f.Provider as provider
from g4f.client import Client


client = Client()
# response = client.images.generate(
#     model="midjourney",
#     prompt="a white siamese cat",
#     response_format="url"
# )
# print(f"Generated image URL: {response.data[0].url}")

response = client.chat.completions.create(
    model="grok-3-r1",
    # provider=provider.Jmuz,
    messages=[{"role": "user", "content": "hello"}],
    web_search=False
)
print(response.choices[0].message.content)
print(response)

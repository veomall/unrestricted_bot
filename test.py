from g4f.client import Client

client = Client()
response = client.images.generate(
    model="midjourney",
    prompt="a white siamese cat",
    response_format="url"
)

print(f"Generated image URL: {response.data[0].url}")

# 90. flux-pro
# 93. dall-e-3
# 94. midjourney

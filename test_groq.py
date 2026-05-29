import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Connect to Groq
from dotenv import load_dotenv
import os
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Testing Groq connection...")
print("-" * 40)

# Send a test message
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are TelecomAI, an expert telecom business analyst."
        },
        {
            "role": "user",
            "content": "Which data package should we promote next month?"
        }
    ],
    max_tokens=300,
    temperature=0.7
)

reply = response.choices[0].message.content
print("Groq is working!")
print("\nAI Response:")
print(reply)
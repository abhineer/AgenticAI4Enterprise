import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from langmem import create_prompt_optimizer

# Load .env file
load_dotenv()   

# Set OpenAI API key from env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load feedback CSV
# Expected columns: system_prompt | user_prompt | response | human_feedback
df = pd.read_csv("feedback_data.csv")

async def main():
    optimizer = create_prompt_optimizer(
        "openai:gpt-4o-mini",   
        kind="prompt_memory"
    )

    trajectories = []
    for _, row in df.iterrows():
        conversation = []
        if pd.notna(row.get("system_prompt")):
            conversation.append({"role": "system", "content": row["system_prompt"]})
        conversation.append({"role": "user", "content": row["user_prompt"]})
        conversation.append({"role": "assistant", "content": row["response"]})

        feedback = row["human_feedback"]
        trajectories.append((conversation, {"feedback": feedback}))

    # Optimize across all feedback
    better_prompt = await optimizer(trajectories, "You are a coding assistant")
    
    print("\n==== Optimized Prompt ====\n")
    print(better_prompt)

# Run async entrypoint
asyncio.run(main())

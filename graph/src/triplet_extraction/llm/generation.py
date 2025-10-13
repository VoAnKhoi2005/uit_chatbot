from openai import OpenAI

def generate_response_gpt_4_1_mini(system_prompt, user_prompt, client):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content
def api(prompt):
    os.environ['SAMBANOVA_API_KY'] = '3625d6ad-e51e-4d62-97df-149d8de8ffe9'
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KY"),
        base_url="https://api.sambanova.ai/v1",
    )

    response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=prompt,
        temperature=0.1,
        top_p=0.1
    )

    return response.choices[0].message.content

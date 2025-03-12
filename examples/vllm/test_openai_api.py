from openai import OpenAI

def main(model_name: str):
    client = OpenAI(api_key="-")
    client.base_url = "http://127.0.0.1:8080/serve/openai/v1"

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hi there, goodman!"}
        ],
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0
    )
    print(f"\n\nChatCompletion:\n\n  {chat_response.choices[0].message.content}")

    comp_response = client.completions.create(
        model=model_name,
        prompt="Hi there, goodman!",
        temperature=1.0,
        max_tokens=256
    )
    print(f"\n\nCompletion:\n\n  {comp_response.choices[0].text}")

    fake_body = {"stream": False, "model": model_name, "prompt": "test"}
    print(f"\n\nModels:\n")
    print('\n\n'.join(map(str, client.models.list(extra_body=fake_body).data)))

    return None

if __name__ == '__main__':
    main(model_name="test_vllm")

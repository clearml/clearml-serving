from openai import OpenAI

def main(model_name: str = "test_vllm"):
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

    print(f"ChatCompletion: {chat_response.choices[0].message.content}")

    comp_response = client.completions.create(
        model=model_name,
        prompt="Hi there, goodman!",
        temperature=1.0,
        max_tokens=256
    )

    print(f"Completion: \n {comp_response.choices[0].text}")

    return None

if __name__ == '__main__':
    main()
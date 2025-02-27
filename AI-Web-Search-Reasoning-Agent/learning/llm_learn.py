from openai import AsyncOpenAI, OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

import asyncio
import json
import os

load_dotenv()



search_tool = TavilySearchResults(max_results=5, search_depth="advanced")

search_web_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for the current information(latest is 2025) on the given query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on web"
                }
            },
            "required": [
                "query"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

reason_tool = {
    "type": "function",
    "function": {
        "name": "analyze_query",
        "description": "Break down and understand the user's question to create better search queries",
        "parameters": {
            "type": "object",
            "properties": {
                "original_question": {
                    "type": "string",
                    "description": "The user's original question"
                },
                "step_by_step_thoughts": {
                    "type": "string",
                    "description": "The AI's reasoning about how to approach the question"
                },
            },
            "required": ["original_question", "step_by_step_thoughts"],
            "additionalProperties": False
        },
        "strict": True
    }
}


tools = [search_web_tool, reason_tool]


user_prompt: str = input()

coding_model = OpenAI(api_key="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")

reason_model = OpenAI(api_key="sk-a0dc63253fca4d7cb89720f47c70bff3", base_url="https://api.deepseek.com")

response = coding_model.chat.completions.create(

    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': user_prompt}],
    tools=tools,
    tool_choice={
    "type": "function",
    "function": {
        "name": "web_search"
        }
    }

    )

tool_call = response.choices[0].message.tool_calls[0]
tool_arg = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

query = tool_arg.get('query')
type(query)

def web_search(query):
    if tool_call.function.name == 'web_search':
        search_result = search_tool.invoke({'query': query})

    return search_result

search_result = web_search(query)

reason = reason_model.chat.completions.create(
    model='deepseek-reasoner',
    messages=[
        {'role': 'user', 'content': user_prompt},
        {'role': 'assistant', 'content': f"Web search results: {json.dumps(search_result)}"},
        {'role': 'user', 'content': 'Based on the search results and your own knowledge based information about the question, please analyze and explain the answer.'}
    ]
)


print("\nWebsites searched by the tool:\n ")
for result in search_result:
    print("\nWebsite:", result['url'])
    print("\nContent snippet:", result['content'])
    print("----------------------------------")


print("\nThinking the answer:\n")
print(reason.choices[0].message.reasoning_content)
print("----------------------------------")
print("\nModel's response based on web search:\n")
print(reason.choices[0].message.content)






from openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

# Load environment variables once at startup
load_dotenv()

# Validate API keys early
openai_api_key = os.environ.get("OPENAI_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY not found in environment variables")
    exit(1)
if not deepseek_api_key:
    print("Error: DEEPSEEK_API_KEY not found in environment variables")
    exit(1)

# Initialize clients once
coding_model = OpenAI(api_key=openai_api_key)
reason_model = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
search_tool = TavilySearchResults(max_results=5, search_depth="advanced")

# Define search web tool
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
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }
}

tools = [search_web_tool]

def clean_markdown(markdown_text):
    """
    Convert markdown-formatted text to plain readable text.
    """
    # Compile regex patterns once for better performance
    code_block_pattern = re.compile(r'```[\w]*\n(.*?)\n```', re.DOTALL)
    inline_code_pattern = re.compile(r'`([^`]+)`')
    header_pattern = re.compile(r'#{1,6}\s+(.*?)(?:\n|$)')
    bold_pattern = re.compile(r'\*\*(.*?)\*\*')
    italic_pattern = re.compile(r'\*(.*?)\*')
    bullet_pattern = re.compile(r'^- (.*?)$', re.MULTILINE)
    numbered_pattern = re.compile(r'^\d+\.\s+(.*?)$', re.MULTILINE)
    newlines_pattern = re.compile(r'\n{3,}')
    links_pattern = re.compile(r'\[(.*?)\]\(.*?\)')
    
    # Apply transformations
    text = code_block_pattern.sub(r'\1', markdown_text)
    text = inline_code_pattern.sub(r'\1', text)
    text = header_pattern.sub(r'\1:\n', text)
    text = bold_pattern.sub(r'\1', text)
    text = italic_pattern.sub(r'\1', text)
    text = bullet_pattern.sub(r'• \1', text)
    text = numbered_pattern.sub(r'• \1', text)
    text = newlines_pattern.sub('\n\n', text)
    text = links_pattern.sub(r'\1', text)
    
    return text.strip()

def web_search(query):
    """Perform web search with the given query"""
    return search_tool.invoke({'query': query})

def main():
    # Get user input
    user_prompt = input().strip()[:500]
    if not user_prompt:
        print('Please enter a valid question.')
        return
    
    # Get search query from OpenAI
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
    tool_arg = json.loads(tool_call.function.arguments)
    query = tool_arg.get('query')
    
    # Execute search
    search_result = web_search(query) if tool_call.function.name == 'web_search' else []
    
    # Use thread pool to run API calls in parallel where possible
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start the reasoning task
        reason_future = executor.submit(
            reason_model.chat.completions.create,
            model='deepseek-reasoner',
            messages=[
                {'role': 'user', 'content': user_prompt},
                {'role': 'assistant', 'content': f"Web search results: {json.dumps(search_result)}"},
                {'role': 'user', 'content': 'Based on the search results and your own knowledge based information about the question, please analyze and explain the answer.'}
            ]
        )
        
        # Wait for reasoning to complete
        reason = reason_future.result()
        reasoning_prompt = reason.choices[0].message.reasoning_content
        
    # Get the final completion
    completion = coding_model.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': f"Web search results: {json.dumps(search_result)}"},
            {'role': 'assistant', 'content': f"Reasoning prompt: {reasoning_prompt}"},
            {'role': 'user', 'content': 'Based on the search results, and your own knowledge based information about the question, please analyze and explain the answer by following the guideline from reasoning prompt. Specify from where you get the informations to construct the answer.'}
        ]
    )
    
    # Process and print results
    clean_output = clean_markdown(completion.choices[0].message.content)
    
    print("\nWebsites searched by the tool:\n")
    for result in search_result:
        print("\nWebsite:", result['url'])
        print("\nContent snippet:", result['content'])
        print("----------------------------------")
    
    print("\nThinking the answer:\n")
    print(reason.choices[0].message.reasoning_content)
    print("----------------------------------")
    print("\nModel's response based on web search:\n")
    print(clean_output)

if __name__ == "__main__":
    main()
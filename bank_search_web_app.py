import os
import re
import sys
import time
import openai
import shutil
import requests
import backoff
import streamlit as st
import html2text
import tiktoken
from googlesearch import search
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.tools import TavilySearchResults

NUM_SEARCH = 2  # Number of links to parse from Google
SEARCH_TIME_LIMIT = 10  # Max seconds to request website sources before skipping to the next URL
TOTAL_TIMEOUT = 20  # Overall timeout for all operations
MAX_CONTENT = 500  # Number of words to add to LLM context for each search result
MAX_TOKENS = 1000 # Maximum number of tokens LLM generates
# LLM_MODEL = 'gpt-4o-mini' #'gpt-3.5-turbo' #'gpt-4o'
LLM_MODEL = 'gpt-4o'
encoding = tiktoken.encoding_for_model('gpt-4o')

# Save markdown content utility
def save_markdown(content, file_path):
    with open(file_path, 'a') as file:
        file.write(content)


def count_tokens(encoding, text):
    return len(encoding.encode(text[0]['content']))


def generate_markdown(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    markdown_content = h.handle(html_content)
    return markdown_content


# Webpage Fetch and Summarization Logic (using your provided functions)
def trace_function_factory(start):
    """Create a trace function to timeout request"""
    def trace_function(frame, event, arg):
        if time.time() - start > TOTAL_TIMEOUT:
            raise TimeoutError('Website fetching timed out')
        return trace_function
    return trace_function


def fetch_webpage(url, timeout):
    """Fetch the content of a webpage given a URL and a timeout."""
    start = time.time()
    sys.settrace(trace_function_factory(start))
    try:
        print(f"Fetching link: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        # soup = BeautifulSoup(response.text, 'lxml')
        page_text = generate_markdown(response.text)
        # paragraphs = soup.find_all('p')
        # page_text = ' '.join([para.get_text() for para in paragraphs])
        return url, page_text
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Error fetching {url}: {e}")
    finally:
        sys.settrace(None)
    return url, None


def parse_google_results(query, num_search=NUM_SEARCH, search_time_limit=SEARCH_TIME_LIMIT):
    """Perform a Google search and parse the content of the top results from a specific site."""
    site_query = f"site:https://www.idfcfirstbank.com/ {query}"
    urls = search(site_query, num_results=num_search)

    max_workers = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_webpage, url, search_time_limit): url for url in urls}
        return {url: page_text for future in as_completed(future_to_url) if (url := future.result()[0]) and (page_text := future.result()[1])}


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def google_check_search(query, file_path, msg_history=None, llm_model="gpt-4"):
    """Check if query requires search and execute Google search."""
    search_dic = parse_google_results(query)
    search_result_md = "\n".join([f"{number+1}. {link}" for number, link in enumerate(search_dic.keys())])
    save_markdown(f"## Sources\n{search_result_md}\n\n", file_path)
    return search_dic


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def tavily_check_search(query, file_path, msg_history=None, llm_model="gpt-4"):

    tool = TavilySearchResults(max_results = 5,
                               search_depth="advanced",
                               include_answer=True,
                               include_raw_content=True,
                               include_images=False,
                               include_domains=["https://idfcfirstbank.com/"])

    search_dic_response = tool.invoke({"query": query})
    search_dic = {entry['url']: entry['content'] for entry in search_dic_response}
    search_result_md = "\n".join([f"{number+1}. {link}" for number, link in enumerate(search_dic.keys())])
    save_markdown(f"## Sources\n{search_result_md}\n\n", file_path)
    return search_dic


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def llm_answer(query, file_path, msg_history=None, search_dic=None, llm_model=LLM_MODEL, max_content=MAX_CONTENT, max_tokens=MAX_TOKENS, debug=False):
    if search_dic:
        context_block = "\n".join([f"[{i+1}]({url}): {content}" for i, (url, content) in enumerate(search_dic.items())])
        prompt = cited_answer_prompt.format(context_block=context_block, query=query)
        system_prompt = system_prompt_cited_answer
    else:
        prompt = answer_prompt.format(query=query)
        system_prompt = system_prompt_answer

    msg_history = msg_history or []
    new_msg_history = msg_history + [{"role": "user", "content": prompt}]

    total_tokens = count_tokens(encoding, new_msg_history) + len(system_prompt.split())
    if total_tokens > 128000:  # Adjust this limit as per your model's max tokens
        print("Warning: Exceeding maximum token limit. Reducing message history.")
        # Truncate the history or adjust as needed
        while total_tokens > 128000 and msg_history:
            msg_history.pop(0)  # Remove the oldest message
            total_tokens = token_count(msg_history) + len(system_prompt.split())
    
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": system_prompt}, *new_msg_history],
            max_tokens=max_tokens,
            stream=True
        )

        # print("\n" + "*" * 20 + " LLM START " + "*" * 20)
        save_markdown(f"## Answer\n", file_path)
        content = []
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                content.append(chunk_content)
                # print(chunk_content, end="")
                save_markdown(chunk_content, file_path)

        # print("\n" + "*" * 21 + " LLM END " + "*" * 21 + "\n")
        save_markdown("\n\n", file_path)
        new_msg_history = new_msg_history + [{"role": "assistant", "content": ''.join(content)}]

    except openai.BadRequestError as e:
        print(f"Error: {e}")  # Log the error if needed
        save_markdown("## Unable to find relevant results\n", file_path)
        return new_msg_history + [{"role": "assistant", "content": "Unable to find relevant results."}]

    return new_msg_history


system_prompt_answer = """You are a helpful assistant who is expert at answering user's queries"""
answer_prompt = """Generate a response that is informative and relevant to the user's query
User Query:
{query}
"""

system_prompt_cited_answer = """You are a helpful assistant who is expert at answering user's queries based on the cited context for IDFC First Bank."""
cited_answer_prompt = """
Provide a relevant, informative response to the user's query using the given context (search results with [citation number](website link) and brief descriptions).
- Answer directly without referring the user to any external links.
- Use an unbiased, journalistic tone and avoid repeating text.
- Format your response in markdown with bullet points for clarity.
- If the exact result is not available based on the provided context (e.g., when the fixed deposit tenure is for 1 year and the query is for 366 or 380 days), return the closest matching result.
- If no relevant result is found, inform the user that there are no results for their query.
- Cite all information using [citation number](website link) notation, matching each part of your answer to its source.

Context Block:
{context_block}

User Query:
{query}
"""


st.set_page_config(page_title="IDFC First Bank Search", page_icon="🔍")

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("IDFC FIRST Bank Logo.jpg", width=200)

openai.api_key = st.secrets["API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

client = openai.OpenAI(api_key = st.secrets["API_KEY"])

left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.markdown("<h2 style='text-align: center;'>Search</h2>", unsafe_allow_html=True)

query = st.text_input("Type your search query here:", "", key="search", 
                          placeholder="Search...", 
                          label_visibility='collapsed', 
                          max_chars=100, 
                          help="Enter your search terms")

if query:
    with st.spinner("Searching the web..."):
        file_path = "search_results.md"
        search_results = None

        search_results = tavily_check_search(query, file_path)
        
        if not search_results:  # If search_results is empty or None
            search_results = google_check_search(query, file_path)

        # Summarize results using the LLM
        msg_history = llm_answer(query, file_path, search_dic=search_results)

        # Display the summarized answer
        st.markdown(f"**Answer**: {msg_history[-1]['content']}")

        search_result_md = "\n".join([f"{number + 1}. {link}" for number, link in enumerate(search_results.keys())])
        st.markdown(f"## Sources\n{search_result_md}\n\n")
else:
    st.warning("Please enter a search term.")

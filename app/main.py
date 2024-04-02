import os

IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'
MAX_SIZE = int(os.environ.get('MAX_SIZE', '3'))
MAX_LEN = int(os.environ.get('MAX_LEN', '512'))
TOP_K_BM25 = int(os.environ.get('TOP_K_BM25', '10'))
MODEL = os.environ.get('MODEL', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO')
ENABLE_EMBEDDING = os.environ.get('ENABLE_EMBEDDING', 'false') == 'true'
MODEL_EMBEDDING = os.environ.get('MODEL_EMBEDDING', 'thenlper/gte-small')
TOP_K_EMBEDDING = int(os.environ.get('TOP_K', '5'))

from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from playwright.async_api import async_playwright
from collections import defaultdict
from PIL import Image
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModel
from tree_sitter_languages import get_language, get_parser
from tree_sitter import Node
from rank_bm25 import BM25Okapi
import torch.nn.functional as F
import torch
import numpy as np
import re
import io
import json
import playwright
import asyncio
import threading
import concurrent.futures


client = InferenceClient(model=f'https://api-inference.huggingface.co/models/{MODEL}')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
special_tokens = set(tokenizer.all_special_tokens)

language = get_language('html')
parser = get_parser('html')


def chunk_node(node: Node, text: str, max_chars: int = 1500):
    chunks = []
    current_chunk = ""
    for child in node.children:
        if child.end_byte - child.start_byte > max_chars:
            chunks.append(current_chunk)
            current_chunk = ""
            chunks.extend(chunk_node(child, text, max_chars))
        elif child.end_byte - child.start_byte + len(current_chunk) > max_chars:
            chunks.append(current_chunk)
            current_chunk = text[child.start_byte: child.end_byte]
        else:
            current_chunk += text[child.start_byte: child.end_byte]
    chunks.append(current_chunk)

    return chunks


def chunking(text, max_len=1500, min_len=20):
    tree = parser.parse(bytes(text, 'utf-8'))
    node = tree.root_node

    chunks = []
    current_chunk = ''
    for child in node.children:
        if child.end_byte - child.start_byte > max_len:
            chunks.append(current_chunk)
            current_chunk = ''
            chunks.extend(chunk_node(child, text, max_len))
        elif child.end_byte - child.start_byte + len(current_chunk) > max_len:
            chunks.append(current_chunk)
            current_chunk = text[child.start_byte: child.end_byte]
        else:
            current_chunk += text[child.start_byte: child.end_byte]

    chunks.append(current_chunk)
    chunks = [c.strip() for c in chunks]
    chunks = [c for c in chunks if len(c) >= min_len]
    return chunks


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Embedding:
    model = None
    tokenizer = None

    def initialize(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained(MODEL_EMBEDDING)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_EMBEDDING)

    def encode(self, strings):
        self.initialize()
        batch_dict = self.tokenizer(
            strings,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T)
        return scores[0].tolist()


class ConnectionManager:
    def __init__(self, max_size=10):
        self.active_connections: list[WebSocket] = []
        self.browser = {}
        self.page = {}
        self.done = {}
        self.executing = {}
        self.queue = {}
        self.status = defaultdict(list)
        self.max_size = max_size
        self.playwright = None

    async def initialize_playwright(self):
        if self.playwright is None:
            self.playwright = await async_playwright().start()

    async def connect(self, websocket: WebSocket, client_id):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.queue[client_id] = len(self.queue)

    async def disconnect(self, websocket: WebSocket, client_id):
        await websocket.close()
        self.active_connections.remove(websocket)
        if client_id in self.browser:
            await self.browser[client_id].close()
            self.browser.pop(client_id, None)
            self.page.pop(client_id, None)
            self.done.pop(client_id, None)
            self.executing.pop(client_id, None)
            self.queue.pop(client_id, None)
            self.status.pop(client_id, None)

        for k in self.queue.keys():
            self.queue[k] -= 1

    async def initialize(self, client_id):
        browser = await self.playwright.chromium.launch()
        page = await browser.new_page()
        self.browser[client_id] = browser
        self.page[client_id] = page
        await self.page[client_id].goto('https://playwright.dev')
        self.done[client_id] = True
        print(f'done initialize {client_id}')

    async def send_personal_message(self, message: dict, websocket: WebSocket, client_id):
        message['text'] = message['text'].replace('\n', '<br>')
        if self.done.get(client_id, False):
            url = self.page[client_id].url
        else:
            url = None
        message['url'] = url
        await websocket.send_text(json.dumps(message))


manager = ConnectionManager(max_size=MAX_SIZE)
embedding = Embedding()


class Command(BaseModel):
    command: str
    client_id: str


app = FastAPI()

# https://github.com/lavague-ai/LaVague/blob/main/src/lavague/prompts.py
system_prompt = """
Your goal is to write Async Playwright code to answer queries.

Your answer must be a Python markdown only.
You can have access to external websites and libraries.

You can assume the following code has been executed:
```python
from playwright.async_api import async_playwright
playwright = await async_playwright().start()
browser = await playwright.chromium.launch()
page = await browser.new_page()
```

---

HTML:
<!DOCTYPE html>
<html>
<head>
    <title>Mock Search Page</title>
</head>
<body>
    <h1>Search Page Example</h1>
    <input id="searchBar" type="text" placeholder="Type here to search...">
    <button id="searchButton">Search</button>
    <script>
        document.getElementById('searchButton').onclick = function() {{
            var searchText = document.getElementById('searchBar').value;
            alert("Searching for: " + searchText);
        }};
    </script>
</body>
</html>

Query: Click on the search bar 'Type here to search...', type 'selenium', and press the 'Enter' key

Completion:
```python
# Let's proceed step by step.
# First we need to identify the component first, then we can click on it.

# Based on the HTML, the link can be uniquely identified using the ID "searchBar"
# Let's use this ID with Selenium to identify the link
search_bar = await page.query_selector("#searchBar")

# Now we can type the asked input
await search_bar.type("selenium")

# Press the 'Enter' key
await search_bar.press("Enter")
```

---

HTML:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mock Page for Selenium</title>
</head>
<body>
    <h1>Welcome to the Mock Page</h1>
    <div id="links">
        <a href="#link1" id="link1">Link 1</a>
        <br>
        <a href="#link2" class="link">Link 2</a>
        <br>
    </div>
</body>
</html>

Query: Click on the title Link 1 and then click on the title Link 2

Completion:
```python
# Let's proceed step by step.
# First we need to identify the first component, then we can click on it. Then we can identify the second component and click on it.

# Click on the link with text "Link 1"
link1 = await page.get_by_text("Link 1")
await link1.click()

# Click on the link with text "Link 2"
link2 = await page.get_by_text("Link 2")
await link2.click()
```

---

HTML:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enhanced Mock Page for Selenium Testing</title>
</head>
<body>
    <h1>Enhanced Test Page for Selenium</h1>
    <div class="container">
        <button id="firstButton" onclick="alert('First button clicked!');">First Button</button>
        <!-- This is the button we're targeting with the class name "action-btn" -->
        <button class="action-btn" onclick="alert('Action button clicked!');">Action Button</button>
        <div class="nested-container">
            <button id="testButton" onclick="alert('Test Button clicked!');">Test Button</button>
        </div>
        <button class="hidden" onclick="alert('Hidden button clicked!');">Hidden Button</button>
    </div>
</body>
</html>

Query: Click on the Button 'First Button'

Completion:
```python
# Let's proceed step by step.
# First we need to identify the button first, then we can click on it.

# Based on the HTML provided, we need to devise the best strategy to select the button.
# The action button can be identified using the class name "action-btn"
first_button = await page.query_selector("#firstButton")

# Then we can click on it
await first_button.click()
```

---

HTML:
<!DOCTYPE html>
<html lang="en">
{context_str}
</html>

Query: {query_str}

Completion:
```python
# Let's proceed step by step.
"""


async def video_streamer(client_id):
    f = 'black.jpg'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f, 'rb') as fopen:
        black_frame = fopen.read()
    while True:
        if client_id in manager.done:
            r = await manager.page[client_id].screenshot()
            image = Image.open(io.BytesIO(r))
            image = image.convert('RGB')
            jpeg_bytes = io.BytesIO()
            image.save(jpeg_bytes, format='JPEG')
            frame = jpeg_bytes.getvalue()
        else:
            frame = black_frame

        yield (
            b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

        await asyncio.sleep(0.01)


async def run_command(command, client_id):
    if not manager.done.get(client_id, False):
        manager.status[client_id].append('please wait im initializing')
        return

    if manager.executing.get(client_id, False):
        manager.status[client_id].append('<b>hey im running something, please wait</b>\n')
    else:
        manager.executing[client_id] = True
        splitted = command.split('\n')
        splitted = [s for s in splitted if len(s) > 2]
        for command_ in splitted:
            manager.status[client_id].append('reading html\n')
            html = await manager.page[client_id].content()
            manager.status[client_id].append('done reading html\n')

            chunks = chunking(text=html, max_len=MAX_LEN)
            tokenized_corpus = [doc.split(' ') for doc in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = command.split(' ')
            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[-TOP_K_BM25:]
            top_chunks = [chunks[i] for i in top_indices]

            if ENABLE_EMBEDDING:
                manager.status[client_id].append('converting to embedding\n')
                embedding_score = embedding.encode([command] + top_chunks)
                top_indices = np.argsort(doc_scores)[-TOP_K_EMBEDDING:]
                top_chunks = [chunks[i] for i in top_indices]
                manager.status[client_id].append('done convert to embedding\n')

            html = '\n'.join(top_chunks[::-1])
            print(html)

            gen_input = system_prompt.format(context_str=html, query_str=command_)

            try:

                r = client.text_generation(
                    prompt=gen_input,
                    max_new_tokens=1024,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    stream=True,
                    stop_sequences=['```']
                )
                all_texts = []
                for r_ in r:
                    if r_ in special_tokens:
                        continue
                    manager.status[client_id].append(r_)
                    all_texts.append(r_)
                manager.status[client_id].append('\n')

                all_texts = ''.join(all_texts).replace('```', '')
                all_texts = all_texts.replace(
                    'page.', f"manager.page['{client_id}'].")

                exec(
                    f'async def __ex(): ' +
                    ''.join(f'\n {l}' for l in all_texts.split('\n'))
                )
                await locals()['__ex']()

            except Exception as e:
                e = f'error: {str(e)}\n'
                manager.status[client_id].append(e)

            await asyncio.sleep(2.0)

        manager.executing[client_id] = False


@app.get('/')
async def get():
    f = 'index.html'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f) as fopen:
        html = fopen.read()
    return HTMLResponse(html)


@app.post('/command')
async def get_command(command: Command, background_tasks: BackgroundTasks = None):
    command = command.dict()
    background_tasks.add_task(run_command, **command)


@app.get('/queue')
async def get():
    return manager.queue


@app.get('/video_feed')
async def video_feed(client_id: str):
    return StreamingResponse(
        video_streamer(client_id=client_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.websocket('/ws/{client_id}')
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.initialize_playwright()
    await manager.connect(websocket, client_id=client_id)
    try:
        while True:
            if manager.queue[client_id] >= manager.max_size:
                d = {
                    'flush': True,
                    'text': f'your current queue is {(manager.queue[client_id] - manager.max_size) + 1}'}
                await manager.send_personal_message(d, websocket, client_id)
            else:
                if client_id in manager.done:
                    if len(manager.status[client_id]):
                        t = manager.status[client_id].pop(0)
                        d = {
                            'flush': False,
                            'text': t
                        }
                        await manager.send_personal_message(d, websocket, client_id)

                else:
                    d = {
                        'flush': False,
                        'text': 'initializing browser ..\n'
                    }
                    await manager.send_personal_message(d, websocket, client_id)
                    await manager.initialize(client_id)
                    manager.status[client_id].append('done initialized browser\n')
                    manager.status[client_id].append('give me something\n\n')

            await asyncio.sleep(0.001)
            await websocket.send_text('')

    except Exception as e:
        await manager.disconnect(websocket, client_id)

    except WebSocketDisconnect:
        await manager.disconnect(websocket, client_id)

import os

IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'
MAX_SIZE = int(os.environ.get('MAX_SIZE', '3'))
MODEL = os.environ.get('MODEL', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO')

from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from playwright.async_api import async_playwright
from collections import defaultdict
from PIL import Image
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import io
import json
import playwright
import asyncio
import threading
import concurrent.futures


client = InferenceClient(model=f'https://api-inference.huggingface.co/models/{MODEL}')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
special_tokens = set(tokenizer.all_special_tokens)


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


class Command(BaseModel):
    command: str
    client_id: str


app = FastAPI()

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
search_bar = await page.get_by_id("searchBar")
await search_bar.click()

# Now we can type the asked input
await search_bar.fill("selenium")

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

        await asyncio.sleep(0.05)


async def run_command(command, client_id):
    if not manager.done.get(client_id, False):
        manager.status[client_id].append('please wait im initializing')
        return

    if manager.executing.get(client_id, False):
        manager.status[client_id].append('<b>hey im running something, please wait</b>\n')
    else:
        manager.executing[client_id] = True
        gen_input = system_prompt.format(query_str=command)

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

        all_texts = ''.join(all_texts).replace('```', '')
        all_texts = all_texts.replace(
            'page.', f"manager.page['{client_id}'].")

        try:
            exec(
                f'async def __ex(): ' +
                ''.join(f'\n {l}' for l in all_texts.split('\n'))
            )
            await locals()['__ex']()
        except Exception as e:
            e = f'error: {str(e)}'
            manager.status[client_id].append(e)

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

import sys
import os

IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'
MAX_SIZE = int(os.environ.get('MAX_SIZE', '3'))

if IMPORT_LOCAL:
    SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    sys.path.insert(0, SOURCE_DIR)

from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from playwright.async_api import async_playwright
from collections import defaultdict
from PIL import Image
import io
import json
import playwright
import asyncio
import time


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

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))


manager = ConnectionManager(max_size=MAX_SIZE)


class Command(BaseModel):
    command: str
    client_id: str


app = FastAPI()

system_prompt = """
you are Async Playwright Python agent, you always response with Python code only, assumed `browser` dan `page` already initialized, you just need to continue using `page` variable, and no need to close.
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


@app.get('/')
async def get():
    f = 'index.html'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f) as fopen:
        html = fopen.read()
    return HTMLResponse(html)


@app.post('/command')
async def get_command(command: Command):
    print(command)


@app.get('/queue')
async def get():
    return manager.queue


@app.get('/video_feed')
async def video_feed(client_id: str):
    print(client_id)
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
                await manager.send_personal_message(d, websocket)
            else:
                if client_id in manager.done:
                    if len(manager.status[client_id]):
                        t = manager.status[client_id].pop(0)
                        d = {
                            'flush': False,
                            'text': t
                        }
                        await manager.send_personal_message(d, websocket)

                else:
                    d = {
                        'flush': False,
                        'text': 'initializing browser ..'
                    }
                    await manager.send_personal_message(d, websocket)
                    await manager.initialize(client_id)
                    manager.status[client_id].append('done initialized browser')
                    manager.status[client_id].append('give me something')

            await asyncio.sleep(0.05)
            await websocket.send_text('')

    except Exception as e:
        await manager.disconnect(websocket, client_id)

    except WebSocketDisconnect:
        await manager.disconnect(websocket, client_id)

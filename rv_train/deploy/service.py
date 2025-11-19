# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import base64
import io
import json
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from rv_train.deploy.data_models import So100Base64DataModel
from rv_train.deploy.model_manager import So100ModelManager

rbv_mm = None

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def rgb_from_base64(base64_string: str) -> np.ndarray:
    img = base64.b64decode(base64_string)
    array_bytes = io.BytesIO(img)
    return np.load(array_bytes)


@app.post("/predict_base64")
async def predict(data: So100Base64DataModel):
    image_data = np.array([rgb_from_base64(d_rgb) for d_rgb in data.base64_rgb])
    state_data = np.array(data.state)
    instr_data = data.instr
    start_time = time.time()
    with torch.no_grad():
        output, _ = rbv_mm.forward(image_data, state_data, instr_data)
    print(f"Time taken: {time.time() - start_time}")
    return output


@app.post("/predict_base64_stream")
async def predict_base64_stream(data: So100Base64DataModel):
    image_data = np.array([rgb_from_base64(d_rgb) for d_rgb in data.base64_rgb])
    state_data = np.array(data.state)
    instr_data = data.instr

    def generate():
        start_time = time.time()
        assert rbv_mm.cfg.EXP.MODEL == "qwen"
        last_action_txt = ""
        for i in range(rbv_mm.cfg.MODEL.QWEN.horizon):
            with torch.no_grad():
                output, last_action_txt = rbv_mm.forward(
                    image_data,
                    state_data,
                    instr_data,
                    get_one_step_action=True,
                    last_action_txt=last_action_txt,
                )
                print(last_action_txt)
            print(f"Time taken: {time.time() - start_time}")
            yield json.dumps({"index": i, "value": output}) + "\n"
        yield json.dumps({"time_taken": time.time() - start_time}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def get_ip_address():
    import socket

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


if __name__ == "__main__":
    rbv_mm = So100ModelManager()
    PORT = 10000
    print()
    print(f"IP address: {get_ip_address()}")
    print(f"Go to http://{get_ip_address()}:{PORT}/docs for the API documentation")
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT)

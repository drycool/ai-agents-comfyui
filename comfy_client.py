"""ComfyUI API Client for AI Agents System."""
import os
import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://localhost:8188")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")


class ComfyAPIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(self, host: str = None):
        self.host = host or COMFYUI_HOST
        self.client_id = f"agent_{int(time.time())}"
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def queue_prompt(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"prompt": workflow, "client_id": self.client_id}
        response = requests.post(f"{self.host}/prompt", json=payload)
        response.raise_for_status()
        return response.json()

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        response = requests.get(f"{self.host}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    def get_queue(self) -> Dict[str, Any]:
        response = requests.get(f"{self.host}/queue")
        response.raise_for_status()
        return response.json()

    def get_system_stats(self) -> Dict[str, Any]:
        response = requests.get(f"{self.host}/system_stats")
        response.raise_for_status()
        return response.json()

    def get_images(self, prompt_id: str) -> List[Dict[str, Any]]:
        history = self.get_history(prompt_id)
        if prompt_id not in history:
            return []
        images = []
        outputs = history[prompt_id].get("outputs", {})
        for node_id, node_data in outputs.items():
            if "images" in node_data:
                for img in node_data["images"]:
                    images.append({"node_id": node_id, "filename": img["filename"], "type": img.get("type", "output"), "subfolder": img.get("subfolder", "")})
        return images

    def download_image(self, image_info: Dict[str, Any], output_path: str = None) -> str:
        filename = image_info["filename"]
        img_type = image_info.get("type", "output")
        subfolder = image_info.get("subfolder", "")
        parts = [self.host, "view"]
        if subfolder:
            parts.extend(["", subfolder])
        parts.extend(["", filename])
        url = "/".join(parts)
        response = requests.get(url)
        response.raise_for_status()
        if output_path is None:
            output_path = Path(OUTPUT_DIR) / filename
        else:
            output_path = Path(output_path)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return str(output_path)

    def wait_for_completion(self, prompt_id: str, timeout: int = 300, poll_interval: float = 1.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False):
                    return True
                if status.get("errored", False):
                    raise RuntimeError(f"Prompt execution failed")
            time.sleep(poll_interval)
        return False

    def execute_workflow(self, workflow: Dict[str, Any], wait: bool = True, timeout: int = 300) -> Dict[str, Any]:
        result = self.queue_prompt(workflow)
        prompt_id = result.get("prompt_id")
        if not wait:
            return result
        if not self.wait_for_completion(prompt_id, timeout):
            return {"prompt_id": prompt_id, "status": "timeout"}
        images = self.get_images(prompt_id)
        return {"prompt_id": prompt_id, "status": "completed", "images": images}


def create_simple_workflow(prompt: str, negative_prompt: str = "", width: int = 512, height: int = 512, steps: int = 20, cfg: float = 7.0, seed: int = None, checkpoint: str = "v1-5-pruned-emaonly-fp16.safetensors") -> Dict[str, Any]:
    """Create a simple SD1.5 text-to-image workflow."""
    if seed is None:
        seed = int(time.time()) % 1000000
    return {
        "3": {"inputs": {"text": prompt, "clip": ["9", 1]}, "class_type": "CLIPTextEncode", "widgets_values": [prompt]},
        "4": {"inputs": {"text": negative_prompt, "clip": ["9", 1]}, "class_type": "CLIPTextEncode", "widgets_values": [negative_prompt]},
        "6": {"inputs": {"samples": ["10", 0], "vae": ["9", 2]}, "class_type": "VAEDecode", "widgets_values": []},
        "8": {"inputs": {"filename_prefix": "AI_Agent", "images": ["6", 0]}, "class_type": "SaveImage", "widgets_values": ["AI_Agent"]},
        "9": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple", "widgets_values": [checkpoint]},
        "10": {"inputs": {"model": ["9", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["11", 0], "denoise": 1.0}, "class_type": "KSampler", "widgets_values": [seed, steps, cfg, "euler", "normal", 1.0]},
        "11": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage", "widgets_values": [width, height, 1]}
    }


create_simple_sdxl_workflow = create_simple_workflow


if __name__ == "__main__":
    client = ComfyAPIClient()
    try:
        stats = client.get_system_stats()
        print("ComfyUI connected!")
        print(f"Version: {stats.get('system', {}).get('comfyui_version', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")

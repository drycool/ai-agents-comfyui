"""Orchestrator Agent for AI Agents System.

Main orchestration logic that coordinates between:
- Ollama (local LLM for analysis)
- Claude CLI (prompt enhancement)
- Gemini CLI (style suggestions)
- ComfyUI API (image generation)
"""
import os
import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from comfy_client import ComfyAPIClient, create_simple_sdxl_workflow
from cli_wrappers import (
    OllamaWrapper,
    ClaudeWrapper,
    GeminiWrapper,
    check_all_connections
)

load_dotenv()

# Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
WORKFLOWS_DIR = os.getenv("WORKFLOWS_DIR", "./workflows")


class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates AI tools for image generation.

    Flow:
    1. Receive user request
    2. Analyze with Ollama
    3. Get style suggestions from Gemini (optional)
    4. Enhance prompt with Claude (optional)
    5. Generate workflow
    6. Execute in ComfyUI
    7. Verify result with Ollama
    """

    def __init__(self):
        self.ollama = OllamaWrapper()
        self.claude = ClaudeWrapper()
        self.gemini = GeminiWrapper()
        self.comfy = ComfyAPIClient()

        # Ensure directories exist
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(WORKFLOWS_DIR).mkdir(parents=True, exist_ok=True)

    def analyze_request(self, user_request: str) -> Dict[str, Any]:
        """
        Analyze user request using Ollama.

        Args:
            user_request: Natural language request

        Returns:
            Dict with analysis results
        """
        prompt = f"""Analyze this image generation request: "{user_request}"

Extract and return as JSON with fields: subject, style, mood, colors, lighting, composition, technical (width/height/steps).
Return ONLY valid JSON, no other text."""

        try:
            response = self.ollama.run(prompt, format_json=True)

            # Parse JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                # Ensure technical has SD1.5 compatible values
                if "technical" in result:
                    result["technical"].setdefault("width", 512)
                    result["technical"].setdefault("height", 512)
                return result

        except Exception as e:
            print(f"Warning: Failed to parse analysis JSON: {e}")

        # Fallback: return basic analysis (SD1.5 compatible)
        return {
            "subject": user_request,
            "style": "general",
            "mood": "neutral",
            "colors": "natural",
            "lighting": "natural",
            "composition": "centered",
            "technical": {"width": 512, "height": 512, "steps": 20}
        }

    def get_style_suggestions(self, request: str) -> str:
        """
        Get style suggestions from Gemini.

        Args:
            request: User request

        Returns:
            Style suggestions string
        """
        try:
            return self.gemini.suggest_style(request)
        except Exception as e:
            print(f"Warning: Gemini style suggestion failed: {e}")
            return ""

    def enhance_prompt(self, base_prompt: str, style: str = None) -> str:
        """
        Enhance prompt using Claude.

        Args:
            base_prompt: Base prompt to enhance
            style: Optional style description

        Returns:
            Enhanced prompt
        """
        try:
            return self.claude.enhance_prompt(base_prompt, style)
        except Exception as e:
            print(f"Warning: Claude prompt enhancement failed: {e}")
            return base_prompt

    def generate_workflow(self, prompt: str, negative_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate ComfyUI workflow.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            **kwargs: Additional parameters (width, height, steps, cfg, seed)

        Returns:
            ComfyUI workflow dict
        """
        return create_simple_sdxl_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            steps=kwargs.get("steps", 20),
            cfg=kwargs.get("cfg", 7.0),
            seed=kwargs.get("seed")
        )

    def process_simple(self, user_request: str) -> Dict[str, Any]:
        """
        Simple generation pipeline: Ollama -> ComfyUI.

        Args:
            user_request: User request

        Returns:
            Result dict with images
        """
        # Analyze request
        analysis = self.analyze_request(user_request)

        # Build prompt from analysis - ensure all parts are strings
        def ensure_str(val):
            if isinstance(val, list):
                return ", ".join(str(v) for v in val)
            return str(val) if val else ""

        prompt_parts = [
            ensure_str(analysis.get("subject", "")),
            ensure_str(analysis.get("style", "")),
            ensure_str(analysis.get("mood", "")),
            ensure_str(analysis.get("lighting", "")),
            ensure_str(analysis.get("colors", ""))
        ]
        prompt = ", ".join([p for p in prompt_parts if p])

        # Generate
        workflow = self.generate_workflow(
            prompt=prompt,
            width=analysis.get("technical", {}).get("width", 1024),
            height=analysis.get("technical", {}).get("height", 1024),
            steps=analysis.get("technical", {}).get("steps", 20)
        )

        # Execute
        result = self.comfy.execute_workflow(workflow)

        return {
            "status": result.get("status", "unknown"),
            "prompt_id": result.get("prompt_id"),
            "analysis": analysis,
            "prompt": prompt,
            "images": result.get("images", [])
        }

    def process_enhanced(self, user_request: str) -> Dict[str, Any]:
        """
        Enhanced pipeline: Ollama -> Gemini -> Claude -> ComfyUI.

        Args:
            user_request: User request

        Returns:
            Result dict with images
        """
        # Step 1: Analyze with Ollama
        analysis = self.analyze_request(user_request)

        # Step 2: Get style from Gemini
        style_suggestions = self.get_style_suggestions(user_request)

        # Step 3: Build and enhance prompt
        base_prompt = f"{analysis.get('subject', '')}, {analysis.get('style', '')}, {analysis.get('mood', '')}"
        enhanced_prompt = self.enhance_prompt(base_prompt, style_suggestions)

        # Step 4: Generate with ComfyUI
        workflow = self.generate_workflow(
            prompt=enhanced_prompt,
            negative_prompt="low quality, blurry, distorted, ugly, bad anatomy",
            width=analysis.get("technical", {}).get("width", 1024),
            height=analysis.get("technical", {}).get("height", 1024),
            steps=analysis.get("technical", {}).get("steps", 20)
        )

        # Execute
        result = self.comfy.execute_workflow(workflow)

        return {
            "status": result.get("status", "unknown"),
            "prompt_id": result.get("prompt_id"),
            "analysis": analysis,
            "style_suggestions": style_suggestions,
            "prompt": enhanced_prompt,
            "images": result.get("images", [])
        }

    def process_full(self, user_request: str) -> Dict[str, Any]:
        """
        Full pipeline with verification: Ollama -> Gemini -> Claude -> ComfyUI -> Ollama.

        Args:
            user_request: User request

        Returns:
            Result dict with images and verification
        """
        # Run enhanced pipeline first
        result = self.process_enhanced(user_request)

        if result.get("status") != "completed":
            return result

        # Verify with Ollama if images were generated
        if result.get("images"):
            # Get first generated image info
            img_info = result["images"][0]
            result["verification"] = {
                "note": "Full verification not implemented - requires image analysis"
            }

        return result


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI AI Orchestrator")
    parser.add_argument("request", nargs="?", help="Image generation request")
    parser.add_argument("--mode", choices=["simple", "enhanced", "full"], default="simple",
                        help="Pipeline mode")
    parser.add_argument("--check", action="store_true", help="Check connections and exit")
    parser.add_argument("--list-models", action="store_true", help="List Ollama models")

    args = parser.parse_args()

    if args.check:
        print("Checking connections...")
        status = check_all_connections()
        for tool, ok in status.items():
            print(f"  {tool}: {'OK' if ok else 'NOT FOUND'}")
        return

    if args.list_models:
        print("Available Ollama models:")
        models = OllamaWrapper.list_models()
        for m in models:
            print(f"  {m['name']} ({m['size']})")
        return

    if not args.request:
        parser.print_help()
        return

    # Initialize orchestrator
    orchestrator = OrchestratorAgent()

    # Run appropriate pipeline
    print(f"Processing request: {args.request}")
    print(f"Mode: {args.mode}")

    if args.mode == "simple":
        result = orchestrator.process_simple(args.request)
    elif args.mode == "enhanced":
        result = orchestrator.process_enhanced(args.request)
    else:
        result = orchestrator.process_full(args.request)

    # Print results
    print(f"\nStatus: {result.get('status')}")
    if result.get("prompt"):
        print(f"Prompt: {result.get('prompt')}")
    if result.get("images"):
        print(f"Generated {len(result['images'])} image(s)")
        for img in result["images"]:
            print(f"  - {img['filename']}")
    if result.get("prompt_id"):
        print(f"Prompt ID: {result['prompt_id']}")


if __name__ == "__main__":
    main()

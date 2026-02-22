"""CLI Wrappers for AI Agents System.

This module provides wrapper functions for calling:
- Ollama (local LLM models)
- Claude CLI (Anthropic)
- Gemini CLI (Google)
"""
import subprocess
import json
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.3")
OLLAMA_ORCHESTRATOR_MODEL = os.getenv("OLLAMA_ORCHESTRATOR_MODEL", "llama3.3")
OLLAMA_LIGHT_MODEL = os.getenv("OLLAMA_LIGHT_MODEL", "phi4")
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "claude")
GEMINI_CLI_PATH = os.getenv("GEMINI_CLI_PATH", "gemini")


class OllamaWrapper:
    """Wrapper for Ollama CLI operations."""

    @staticmethod
    def run(
        prompt: str,
        model: str = None,
        format_json: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Run a prompt through Ollama.

        Args:
            prompt: The prompt to send to the model
            model: Model name (default: from config)
            format_json: If True, request JSON output
            temperature: Sampling temperature (0.0 - 2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Model's response as string
        """
        model = model or OLLAMA_DEFAULT_MODEL

        cmd = ["ollama", "run", model]

        if format_json:
            cmd.extend(["--format", "json"])

        # Add the prompt as last argument
        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                raise RuntimeError(f"Ollama error: {result.stderr}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Ollama request timed out for model {model}")
        except FileNotFoundError:
            raise RuntimeError("Ollama not found. Make sure it's installed and in PATH")

    @staticmethod
    def list_models() -> List[Dict[str, str]]:
        """List available Ollama models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                return []

            # Parse output
            models = []
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        models.append({
                            "name": parts[0],
                            "size": parts[1],
                            "modified": " ".join(parts[2:])
                        })

            return models

        except Exception:
            return []

    @staticmethod
    def pull(model: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for large models
                encoding='utf-8',
                errors='replace'
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def check_connection() -> bool:
        """Check if Ollama is running."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False


class ClaudeWrapper:
    """Wrapper for Claude CLI operations."""

    @staticmethod
    def run(
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 4096
    ) -> str:
        """
        Run a prompt through Claude CLI.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Claude's response as string
        """
        cmd = [CLAUDE_CLI_PATH, "--print"]

        if system_prompt:
            cmd.extend(["--system", system_prompt])

        cmd.extend(["--max-tokens", str(max_tokens)])
        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                # Try to extract error message
                error_msg = result.stderr or "Unknown Claude CLI error"
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TimeoutError("Claude CLI request timed out")
        except FileNotFoundError:
            raise RuntimeError("Claude CLI not found. Make sure it's installed and in PATH")

    @staticmethod
    def enhance_prompt(base_prompt: str, style: str = None) -> str:
        """
        Enhance a prompt for image generation.

        Args:
            base_prompt: Original prompt
            style: Optional style description

        Returns:
            Enhanced prompt
        """
        style_hint = ""
        if style:
            style_hint = f"\nStyle: {style}"

        prompt = f"""Improve this prompt for AI image generation (ComfyUI/SDXL).
Add details about: lighting, composition, colors, atmosphere, mood.
Keep it descriptive but concise.{style_hint}

Original prompt: {base_prompt}

Return only the improved prompt, nothing else."""

        return ClaudeWrapper.run(prompt)

    @staticmethod
    def generate_workflow(description: str, model: str = "SDXL") -> Dict[str, Any]:
        """
        Generate a ComfyUI workflow from description.

        Args:
            description: What to generate
            model: Model type (SDXL, SD15, etc.)

        Returns:
            Workflow dict
        """
        prompt = f"""Create a ComfyUI workflow JSON for {model}.
Task: {description}

Return ONLY valid JSON, no explanations.
Include these nodes:
- CheckpointLoader
- CLIPTextEncode (positive)
- CLIPTextEncode (negative)
- EmptyLatentImage
- KSampler
- VAEDecode
- SaveImage

Use reasonable default values."""

        result = ClaudeWrapper.run(prompt, max_tokens=4096)

        # Try to parse JSON
        try:
            # Find JSON in output
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Could not parse workflow JSON from Claude response")

    @staticmethod
    def check_connection() -> bool:
        """Check if Claude CLI is available."""
        try:
            result = subprocess.run(
                [CLAUDE_CLI_PATH, "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False


class GeminiWrapper:
    """Wrapper for Gemini CLI operations."""

    @staticmethod
    def run(
        prompt: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7
    ) -> str:
        """
        Run a prompt through Gemini CLI.

        Args:
            prompt: The prompt to send
            model: Gemini model to use
            temperature: Sampling temperature

        Returns:
            Gemini's response as string
        """
        # Build command - exact syntax depends on Gemini CLI version
        cmd = [GEMINI_CLI_PATH]

        # Try different argument patterns
        cmd.extend([
            "--model", model,
            "--temperature", str(temperature)
        ])

        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                raise RuntimeError(f"Gemini CLI error: {result.stderr}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TimeoutError("Gemini CLI request timed out")
        except FileNotFoundError:
            raise RuntimeError("Gemini CLI not found. Make sure it's installed and in PATH")

    @staticmethod
    def suggest_style(request: str) -> str:
        """
        Suggest artistic style and references.

        Args:
            request: What the user wants to generate

        Returns:
            Style suggestions
        """
        prompt = f"""For this image request: "{request}"

Suggest:
1. Artistic style (e.g., photorealistic, anime, oil painting, etc.)
2. Color palette
3. Lighting mood
4. 2-3 famous artists for reference

Keep response concise (3-4 sentences)."""

        return GeminiWrapper.run(prompt)

    @staticmethod
    def analyze_image(image_path: str) -> str:
        """
        Analyze an image using Gemini.

        Args:
            image_path: Path to image file

        Returns:
            Analysis description
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = f"Analyze this image in detail. Describe its composition, style, colors, and quality."

        cmd = [GEMINI_CLI_PATH, "analyze", image_path, "--prompt", prompt]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                raise RuntimeError(f"Gemini CLI error: {result.stderr}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TimeoutError("Gemini CLI image analysis timed out")

    @staticmethod
    def check_connection() -> bool:
        """Check if Gemini CLI is available."""
        try:
            result = subprocess.run(
                [GEMINI_CLI_PATH, "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False


# Convenience functions
def call_ollama(prompt: str, model: str = None, format_json: bool = False) -> str:
    """Convenience function to call Ollama."""
    return OllamaWrapper.run(prompt, model, format_json)


def call_claude(prompt: str, system_prompt: str = None) -> str:
    """Convenience function to call Claude CLI."""
    return ClaudeWrapper.run(prompt, system_prompt)


def call_gemini(prompt: str) -> str:
    """Convenience function to call Gemini CLI."""
    return GeminiWrapper.run(prompt)


# Diagnostic function
def check_all_connections() -> Dict[str, bool]:
    """Check connection status of all CLI tools."""
    return {
        "ollama": OllamaWrapper.check_connection(),
        "claude": ClaudeWrapper.check_connection(),
        "gemini": GeminiWrapper.check_connection()
    }


if __name__ == "__main__":
    # Test connections
    print("Checking CLI tool connections...")
    status = check_all_connections()

    for tool, connected in status.items():
        status_str = "OK" if connected else "NOT FOUND"
        print(f"  {tool}: {status_str}")

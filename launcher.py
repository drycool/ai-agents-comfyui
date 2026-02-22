"""Launcher script for ComfyUI AI Agents System."""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import OrchestratorAgent
from cli_wrappers import check_all_connections, OllamaWrapper
from comfy_client import ComfyAPIClient


def print_status():
    """Print connection status of all services."""
    print("=" * 50)
    print("ComfyUI AI Agents System - Status Check")
    print("=" * 50)

    # Check CLI tools
    print("\nCLI Tools:")
    cli_status = check_all_connections()
    for tool, connected in cli_status.items():
        status = "OK" if connected else "NOT FOUND"
        print(f"  {tool}: {status}")

    # Check ComfyUI
    print("\nComfyUI:")
    try:
        comfy = ComfyAPIClient()
        stats = comfy.get_system_stats()
        print(f"  Status: Connected")
        print(f"  VRAM: {stats.get('vram_used', '?')} / {stats.get('vram_total', '?')}")
    except Exception as e:
        print(f"  Status: Not Connected ({e})")
        print("  Run ComfyUI with: python main.py --listen 0.0.0.0 --port 8188")

    # Check Ollama models
    print("\nOllama Models:")
    models = OllamaWrapper.list_models()
    if models:
        for m in models:
            print(f"  - {m['name']} ({m['size']})")
    else:
        print("  No models found")

    print("\n" + "=" * 50)


def run_generation(request: str, mode: str = "simple"):
    """Run image generation."""
    print(f"\nGenerating: {request}")
    print(f"Mode: {mode}")
    print("-" * 40)

    orchestrator = OrchestratorAgent()

    if mode == "simple":
        result = orchestrator.process_simple(request)
    elif mode == "enhanced":
        result = orchestrator.process_enhanced(request)
    else:
        result = orchestrator.process_full(request)

    print(f"\nStatus: {result.get('status')}")

    if result.get("prompt"):
        print(f"\nPrompt:\n{result['prompt']}")

    if result.get("images"):
        print(f"\nGenerated {len(result['images'])} image(s):")
        for img in result["images"]:
            print(f"  - {img['filename']}")
            # Download image
            try:
                comfy = ComfyAPIClient()
                path = comfy.download_image(img)
                print(f"    Saved to: {path}")
            except Exception as e:
                print(f"    (Could not download: {e})")

    if result.get("prompt_id"):
        print(f"\nPrompt ID: {result['prompt_id']}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI AI Agents System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py --status                    # Check all connections
  python launcher.py "a cat sitting on a bench" # Simple generation
  python launcher.py "a sunset" --mode enhanced # Enhanced mode
  python launcher.py --list-models               # List Ollama models
        """
    )

    parser.add_argument("request", nargs="?", help="Image generation request")
    parser.add_argument("--mode", choices=["simple", "enhanced", "full"],
                        default="simple", help="Pipeline mode (default: simple)")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Check connection status")
    parser.add_argument("--list-models", "-l", action="store_true",
                        help="List available Ollama models")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.list_models:
        print("Available Ollama models:")
        models = OllamaWrapper.list_models()
        for m in models:
            print(f"  {m['name']} ({m['size']}) - {m['modified']}")
    elif args.request:
        run_generation(args.request, args.mode)
    else:
        parser.print_help()
        print_status()


if __name__ == "__main__":
    main()

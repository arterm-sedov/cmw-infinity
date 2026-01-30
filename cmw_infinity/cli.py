"""CLI interface for cmw-infinity."""

from __future__ import annotations

import sys

import click

from .server_config import (
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    get_model_config,
    list_available_models,
)
from .server_manager import InfinityServerManager


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CMW Infinity - Infinity server management for embedding/reranker inference."""
    pass


@cli.command()
def setup() -> None:
    """Verify setup and dependencies."""
    click.echo("Setting up cmw-infinity...")

    # Check infinity-emb installation
    try:
        import subprocess

        result = subprocess.run(
            ["infinity_emb", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            click.echo(f"✓ infinity_emb installed: {result.stdout.strip()}")
        else:
            click.echo("⚠ infinity_emb found but returned error")
    except FileNotFoundError:
        click.echo("✗ infinity_emb not found")
        click.echo("  Install with: pip install infinity-emb[torch,optimum]")
        sys.exit(1)
    except Exception as e:
        click.echo(f"⚠ Could not check infinity_emb: {e}")

    # Check GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            click.echo(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            click.echo("⚠ No GPU detected (CPU mode will be used)")
    except ImportError:
        click.echo("⚠ PyTorch not installed")

    # Check requests
    try:
        import requests

        click.echo(f"✓ requests installed ({requests.__version__})")
    except ImportError:
        click.echo("✗ requests not found")
        click.echo("  Install with: pip install requests")
        sys.exit(1)

    click.echo("\n✓ Setup complete!")


@cli.command()
@click.argument("model_key")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't detach)")
def start(model_key: str, foreground: bool) -> None:
    """Start an Infinity server for a model."""
    try:
        config = get_model_config(model_key)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        available = list_available_models()
        click.echo("\nAvailable models:")
        click.echo(f"  Embedding: {', '.join(available['embedding'])}")
        click.echo(f"  Reranker: {', '.join(available['reranker'])}")
        sys.exit(1)

    manager = InfinityServerManager()
    status = manager.get_status(model_key, config)

    if status.is_running:
        click.echo(f"✓ Server '{model_key}' is already running on port {config.port}")
        return

    click.echo(f"Starting Infinity server for '{model_key}'...")
    click.echo(f"  Model: {config.model_id}")
    click.echo(f"  Port: {config.port}")
    click.echo(f"  Estimated memory: {config.memory_gb} GB")

    success = manager.start(model_key, config, background=not foreground)

    if success:
        if foreground:
            click.echo("\n✓ Server stopped")
        else:
            click.echo(f"✓ Server started on port {config.port}")
            click.echo(f"  Use 'cmw-infinity status' to check health")
    else:
        click.echo("✗ Failed to start server", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_key", required=False)
@click.option("--all", "stop_all", is_flag=True, help="Stop all running servers")
def stop(model_key: str | None, stop_all: bool) -> None:
    """Stop an Infinity server."""
    manager = InfinityServerManager()

    if stop_all:
        running = manager.list_running()
        if not running:
            click.echo("No servers are running")
            return

        click.echo(f"Stopping {len(running)} server(s)...")
        for status in running:
            click.echo(f"  Stopping '{status.model_key}'...")
            manager.stop(status.model_key)
        click.echo("✓ All servers stopped")
        return

    if not model_key:
        click.echo("Error: Specify model key or use --all", err=True)
        sys.exit(1)

    try:
        config = get_model_config(model_key)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    status = manager.get_status(model_key, config)
    if not status.pid:
        click.echo(f"Server '{model_key}' is not running")
        return

    click.echo(f"Stopping server '{model_key}'...")
    if manager.stop(model_key):
        click.echo("✓ Server stopped")
    else:
        click.echo("✗ Failed to stop server", err=True)
        sys.exit(1)


@cli.command()
def status() -> None:
    """Check status of all servers."""
    manager = InfinityServerManager()
    running = manager.list_running()

    if not running:
        click.echo("No servers are running")
        return

    click.echo(f"{'Model':<20} {'Type':<10} {'Port':<8} {'Status':<12} {'Uptime'}")
    click.echo("-" * 65)

    for s in running:
        status_str = "✓ running" if s.is_running else "✗ not responding"
        uptime_str = ""
        if s.uptime_seconds:
            minutes = int(s.uptime_seconds // 60)
            hours = minutes // 60
            if hours > 0:
                uptime_str = f"{hours}h {minutes % 60}m"
            else:
                uptime_str = f"{minutes}m"

        model_type = "embedding" if s.model_key in EMBEDDING_MODELS else "reranker"
        click.echo(f"{s.model_key:<20} {model_type:<10} {s.port:<8} {status_str:<12} {uptime_str}")


@cli.command(name="list")
def list_models() -> None:
    """List all available models."""
    available = list_available_models()

    click.echo("Embedding Models:")
    for key in available["embedding"]:
        config = EMBEDDING_MODELS[key]
        click.echo(f"  {key:<25} {config.model_id:<40} {config.memory_gb} GB")

    click.echo("\nReranker Models:")
    for key in available["reranker"]:
        config = RERANKER_MODELS[key]
        click.echo(f"  {key:<25} {config.model_id:<40} {config.memory_gb} GB")

    click.echo("\nUsage:")
    click.echo("  cmw-infinity start <model-key>")


if __name__ == "__main__":
    cli()

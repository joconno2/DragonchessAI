"""
Send a DragonchessAI training summary to Discord via the research assistant bot.

Usage:
    python notify_training.py results/td/best_nightly
    python notify_training.py results/td/best_nightly --label "Nightly best-config"
    python notify_training.py results/td/d2_baseline results/td/d2_lambda_high  # multiple runs
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import discord
from discord import Color, Embed
from dotenv import load_dotenv

ASSISTANT_DIR = Path(__file__).parent.parent / "assistant"
load_dotenv(ASSISTANT_DIR / ".env")


def read_checkpoint(results_dir: Path) -> dict | None:
    latest = results_dir / "latest.json"
    if not latest.exists():
        return None
    with open(latest) as f:
        return json.load(f)


def format_run_summary(name: str, ckpt: dict) -> dict:
    """Extract key stats from a checkpoint into a summary dict."""
    cfg = ckpt.get("config", {})
    hist = ckpt.get("win_rate_history", [])
    best_wr = max((e["win_rate_vs_ab"] for e in hist), default=0)
    latest_wr = hist[-1]["win_rate_vs_ab"] if hist else 0
    elapsed_h = ckpt.get("elapsed_seconds", 0) / 3600

    return {
        "name": name,
        "batches": ckpt.get("total_batches", 0),
        "games": ckpt.get("total_games", 0),
        "elapsed_h": elapsed_h,
        "best_wr": best_wr,
        "latest_wr": latest_wr,
        "depth": cfg.get("td_depth", "?"),
        "lr": cfg.get("lr", "?"),
        "lambd": cfg.get("lambda", "?"),
    }


def build_embed(summaries: list[dict], label: str) -> Embed:
    """Build a Discord embed from one or more run summaries."""
    embed = Embed(
        title=f"🐉 DragonchessAI — {label}",
        color=Color.blue(),
        timestamp=datetime.now(),
    )

    for s in summaries:
        # Trend arrow
        if s["latest_wr"] >= s["best_wr"] - 0.01:
            trend = "📈"
        elif s["latest_wr"] >= s["best_wr"] * 0.9:
            trend = "➡️"
        else:
            trend = "📉"

        value = (
            f"**Best WR vs AB2:** {s['best_wr']:.1%} {trend}\n"
            f"Latest: {s['latest_wr']:.1%} · "
            f"Batches: {s['batches']:,} · "
            f"Games: {s['games']:,}\n"
            f"Config: depth={s['depth']} λ={s['lambd']} lr={s['lr']}\n"
            f"Runtime: {s['elapsed_h']:.1f}h"
        )
        embed.add_field(name=s["name"], value=value, inline=False)

    # Top-level summary
    if summaries:
        best = max(summaries, key=lambda s: s["best_wr"])
        embed.set_footer(text=f"Best overall: {best['name']} at {best['best_wr']:.1%} vs AB2")

    return embed


def build_failure_embed(label: str, reason: str) -> Embed:
    return Embed(
        title=f"⚠️ DragonchessAI — {label}",
        description=reason,
        color=Color.red(),
        timestamp=datetime.now(),
    )


async def send_to_discord(embed: Embed):
    token = os.environ.get("DISCORD_TOKEN")
    owner_id = os.environ.get("DISCORD_USER_ID")
    if not token or not owner_id:
        print("ERROR: DISCORD_TOKEN or DISCORD_USER_ID not set", file=sys.stderr)
        sys.exit(1)

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        try:
            user = await client.fetch_user(int(owner_id))
            await user.send(embed=embed)
        finally:
            await client.close()

    await client.start(token)


def main():
    parser = argparse.ArgumentParser(description="Send training summary to Discord")
    parser.add_argument("results_dirs", nargs="+", help="Path(s) to results directories")
    parser.add_argument("--label", default="Training Report", help="Embed title label")
    parser.add_argument("--dry-run", action="store_true", help="Print summary instead of sending")
    args = parser.parse_args()

    summaries = []
    for d in args.results_dirs:
        p = Path(d)
        ckpt = read_checkpoint(p)
        if ckpt:
            summaries.append(format_run_summary(p.name, ckpt))
        else:
            print(f"WARNING: No checkpoint found in {d}", file=sys.stderr)

    if not summaries:
        embed = build_failure_embed(args.label, "No checkpoints found in any results directory.")
    else:
        embed = build_embed(summaries, args.label)

    if args.dry_run:
        print(f"Title: {embed.title}")
        for field in embed.fields:
            print(f"\n--- {field.name} ---")
            print(field.value)
        if embed.footer:
            print(f"\nFooter: {embed.footer.text}")
        return

    asyncio.run(send_to_discord(embed))
    print(f"Sent training summary to Discord ({len(summaries)} run(s))")


if __name__ == "__main__":
    main()

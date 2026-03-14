# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sakura-qq-bot is a QQ chatbot built on **NapCat** (QQ client) + **AstrBot** (AI bot framework), orchestrated via Docker Compose on macOS. NapCat handles QQ protocol via OneBot v11 reverse WebSocket; AstrBot routes messages and integrates with Google Gemini for AI responses.

## Key Commands

```bash
# Start/stop services
docker compose up -d
docker compose down
docker compose restart            # or restart a single service: docker compose restart astrbot

# View logs
docker compose logs -f astrbot
docker compose logs -f napcat

# Check status
docker compose ps
```

No build, lint, or test commands exist yet — the project is infrastructure-only (Docker Compose + config).

## Architecture

```
QQ ←→ NapCat (QQ client, container) ──WebSocket──→ AstrBot (bot framework, container)
                                                        │
                                                   Google Gemini API
```

- **NapCat** (`mlikiowa/napcat-docker`): QQ login, message relay. Ports: WebUI 6099, WS 3000, HTTP 3001.
- **AstrBot** (`soulter/astrbot`): Message routing, plugin system, AI provider integration. Ports: WebUI 6185, OneBot WS 6199.
- Containers communicate over a `qqbot` Docker bridge network. Auth via shared `NAPCAT_WS_TOKEN`.
- Custom AstrBot plugins go in `./plugins/` (mounted to `/AstrBot/data/plugins`).

## Environment Setup

Copy `.env.example` to `.env` and fill in: `NAPCAT_UID/GID` (use `id -u`/`id -g`), `NAPCAT_WS_TOKEN`, `ASTRBOT_ADMIN_QQ`, `BOT_QQ`, `GEMINI_API_KEY`.

## Data Directories

- `data/astrbot/` — AstrBot config, DB, installed plugins
- `data/napcat/config/` — NapCat configuration
- `data/napcat/qq/` — QQ login state
- `plugins/` — Local custom AstrBot plugins (hot-reloadable via WebUI)

All `data/` contents are gitignored except `.gitkeep` files.

## macOS Docker Note

Use `host.docker.internal` (not `0.0.0.0` or local IP) when NapCat needs to reach AstrBot from inside Docker on macOS. The WebSocket URL must end with `/ws`.

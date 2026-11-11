# Zero

Зеро (Zero) — локальный «мозг» ИИ-компаньона для macOS. Это фоновый Python-сервис на `127.0.0.1`, к которому по WebSocket подключаются клиенты («тела»). Мозг ведёт agent loop, владеет сессиями, памятью, инструментами и подтверждениями.

Проект в разработке. Текущий этап смотри в [ROADMAP.md](ROADMAP.md) и `git log`.

## Документы

- [ZERO_SPEC.md](ZERO_SPEC.md) — спецификация: архитектура, протокол, форматы.
- [ROADMAP.md](ROADMAP.md) — порядок работы: 68 коммитов, 8 майлстоунов.

## Требования

- macOS
- [uv](https://docs.astral.sh/uv/) — Python 3.12 он поставит сам

## Команды

```bash
uv sync                                               # окружение и зависимости
uv run python -m zero                                 # запуск сервиса на 127.0.0.1:47400
curl 127.0.0.1:47400/health                           # проверка, что сервис отвечает
uv run python scripts/ws_client.py                    # ручной клиент протокола (нужен запущенный сервис)
uv run pytest -q                                      # тесты
uv run ruff check . && uv run ruff format --check .   # линт и формат
```

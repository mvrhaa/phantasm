# Zero

Зеро (Zero) — локальный «мозг» ИИ-компаньона для macOS. Это фоновый Python-сервис на `127.0.0.1`, к которому по WebSocket подключаются клиенты («тела»). Мозг ведёт agent loop, владеет сессиями, памятью, инструментами и подтверждениями.

Проект в разработке. Текущий этап смотри в [ROADMAP.md](ROADMAP.md) и `git log`.

## Документы

- [ZERO_SPEC.md](ZERO_SPEC.md) — спецификация: архитектура, протокол, форматы.
- [ROADMAP.md](ROADMAP.md) — порядок работы: 75 коммитов, 9 майлстоунов.

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

## Команда `zero`

После `uv sync` доступна команда `zero`:

```bash
uv run zero serve          # запуск сервиса (то же, что python -m zero)
uv run zero token          # показать токен для клиентов
uv run zero doctor         # проверить папки, конфиг, БД, токен, порт и провайдер
uv run zero config path    # где лежит config.json
```

## Ключ API

Ключи API хранятся **только** в файле `.env` в корне проекта. Он не попадает в git, его нельзя прочитать файловыми инструментами Зеро, а значение ключа заменяется на `***` в логах и в выводе инструментов.

```bash
uv run zero doctor   # строка key: ключ найден (сам ключ не показывается)
```

## Автозапуск (LaunchAgent)

Чтобы Зеро запускался сам после входа в систему:

```bash
uv run python scripts/install_launchagent.py              # поставить (или переставить) и запустить
uv run python scripts/install_launchagent.py --print      # только показать plist
uv run python scripts/install_launchagent.py --uninstall  # остановить и убрать
```

Агент запускает Python этого проекта с `-m zero`. launchd перезапускает его только после падения:
штатный выход (например, если Зеро уже работает) перезапуска не вызывает. Вывод агента и лог Зеро
лежат в `~/Library/Logs/Zero`. Проверка после перезагрузки: `curl 127.0.0.1:47400/health`.

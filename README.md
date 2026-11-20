# Phantasm

Phantasm — локальный «мозг» ИИ-компаньона для macOS. Это фоновый Python-сервис на `127.0.0.1`, к которому по WebSocket подключаются клиенты («тела»). Мозг ведёт agent loop, владеет сессиями, памятью, инструментами и подтверждениями.

Версия 0.1.0: протокол, инструменты с подтверждениями, память, расписания, фоновые агенты и реальная модель — Gemma 4 через Google AI Studio. Дальше по плану — уведомления и Telegram.

## Документы

- [PHANTASM_SPEC.md](PHANTASM_SPEC.md) — спецификация: архитектура, протокол, форматы.
- [docs/protocol.md](docs/protocol.md) — справочник протокола для клиентов: все сообщения с полями и примерами.

## Требования

- macOS
- [uv](https://docs.astral.sh/uv/) — Python 3.12 он поставит сам

## Установка

```bash
git clone https://github.com/mvrhaa/phantasm && cd phantasm
uv sync              # Python 3.12 и зависимости
uv run phantasm doctor   # всё ли на месте
```

## Модель

Провайдер и модель задаются в `config.json` (`uv run phantasm config path` покажет, где он):

```json
"provider": { "name": "google", "model": "gemini-3.5-flash-lite", "small_model": "gemini-3.5-flash-lite" }
```

Пресеты: `google`, `openrouter`, `mistral`, `ollama`. Заголовки сессий и саммари пишет лёгкая модель `small_model` (у `google` по умолчанию `gemma-4-26b-a4b-it`). Без настройки работает `fake` — эхо для разработки и тестов.

## Команды

```bash
uv sync                                               # окружение и зависимости
uv run python -m phantasm                             # запуск сервиса на 127.0.0.1:47400
curl 127.0.0.1:47400/health                           # проверка, что сервис отвечает
uv run python scripts/ws_client.py                    # ручной клиент протокола (нужен запущенный сервис)
uv run pytest -q                                      # тесты
uv run ruff check . && uv run ruff format --check .   # линт и формат
uv run python scripts/gen_protocol_docs.py            # пересобрать docs/protocol.md
```

## Команда `phantasm`

После `uv sync` доступна команда `phantasm`:

```bash
uv run phantasm serve          # запуск сервиса (то же, что python -m phantasm)
uv run phantasm token          # показать токен для клиентов
uv run phantasm doctor         # проверить папки, конфиг, БД, токен, порт и провайдер
uv run phantasm usage          # сколько запросов и токенов ушло на модель
uv run phantasm config path    # где лежит config.json
uv run phantasm eval           # прогнать задачи оценки на твоей модели (настоящие запросы)
```

## Chrome

Phantasm может работать в твоём Chrome (с твоими входами на сайты) в отдельном окне через расширение из папки `extension/`: `chrome://extensions` → Developer mode → Load unpacked → `extension/`, затем в окне расширения вставь вывод `uv run phantasm token`. Подробнее — `extension/README.md`.

## Ключ API

Ключи API хранятся **только** в файле `.env` в корне проекта. Он не попадает в git, его нельзя прочитать файловыми инструментами Phantasm, а значение ключа заменяется на `***` в логах и в выводе инструментов.

```bash
uv run phantasm doctor   # строка key: ключ найден (сам ключ не показывается)
```

## Автозапуск (LaunchAgent)

Чтобы Phantasm запускался сам после входа в систему:

```bash
uv run python scripts/install_launchagent.py              # поставить (или переставить) и запустить
uv run python scripts/install_launchagent.py --print      # только показать plist
uv run python scripts/install_launchagent.py --uninstall  # остановить и убрать
```

Агент запускает Python этого проекта с `-m phantasm`. launchd перезапускает его только после падения:
штатный выход (например, если Phantasm уже работает) перезапуска не вызывает. Вывод агента и лог Phantasm
лежат в `~/Library/Logs/Phantasm`. Проверка после перезагрузки: `curl 127.0.0.1:47400/health`.

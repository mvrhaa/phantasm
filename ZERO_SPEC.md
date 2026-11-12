# Зеро (Zero) — техническая спецификация мозга

> Имя: **Зеро** (в честь нулевого аркана Таро «Шут», The Fool, номер 0), в коде **Zero** (пакет `zero`, `APP_NAME = "Zero"`). Это «мозг» ИИ-компаньона.
> Документ — источник истины для реализации. Если код расходится со спекой, правится либо код, либо спека, но осознанно.

---

## 0. Как работать по этой спеке

- Порядок работы задаёт **`ROADMAP.md`**: коммит за коммитом, этап за этапом. Раздел 14 здесь — краткая сводка этапов. Этап закрыт, когда выполнены его критерии приёмки и `pytest` с `ruff` зелёные.
- Начинай с **протокола (раздел 6)** и **интерфейса провайдера (раздел 8)**: от них зависит всё остальное.
- Реальных LLM-провайдеров **пока нет**. Всё разрабатывается и тестируется на `FakeProvider`. Не добавляй SDK провайдеров, пока об этом не попросят.
- Каждый модуль покрывается тестами. Тесты не ходят в сеть и не трогают реальную файловую систему вне `tmp_path`.
- Если спека где-то неоднозначна, не выдумывай: оставь `TODO(spec):` и спроси.

---

## 1. Что это

Zero — локальный фоновый сервис на macOS, «мозг» ИИ-компаньона. Он:
- принимает сообщения пользователя от клиентов («тел»): десктоп-приложение с персонажем, позже телефон;
- ведёт **agent loop**: модель → вызовы инструментов → ответ;
- владеет **сессиями, историей, памятью, инструментами, правами и подтверждениями**;
- вызывает **действия, которые объявил клиент** (скриншот, клик, смена выражения персонажа), и ждёт их результат;
- стримит клиенту всё, что происходит: текст, вызовы инструментов, запросы подтверждения.

Клиент («тело») ничего не решает. Он показывает и исполняет. Вся ИИ-логика находится в мозге.

### Не цели (для v1)
- UI. У мозга нет окон, только API. Панель настроек появится позже отдельным клиентом.
- Конкретные LLM-провайдеры. Есть только интерфейс и Fake.
- Мультипользовательность и сетевой доступ. Только `127.0.0.1`, один пользователь.
- Windows и Linux. Код пишется переносимо, но тестируется и пакуется только под macOS.

---

## 2. Стек

| Что | Чем |
|---|---|
| Язык | Python **3.12** |
| Пакеты / окружение | `uv` (`pyproject.toml`, `uv.lock`) |
| HTTP + WebSocket | **FastAPI** + `uvicorn[standard]` |
| Модели данных / валидация | **pydantic v2** |
| БД | **SQLite** через `aiosqlite`, миграции простыми SQL-файлами с таблицей `schema_version` |
| HTTP-клиент (web-инструменты) | `httpx` (async) |
| Извлечение текста со страниц | `trafilatura` или `selectolax` (выбрать один) |
| Планировщик (этап 6) | свой на asyncio, без внешних зависимостей |
| Тесты | `pytest`, `pytest-asyncio`, `httpx` / `starlette.testclient` для WS |
| Линт / формат | `ruff` (line-length 100) |
| Типы | аннотации везде, `mypy --strict` по возможности (не блокирует) |

Всё асинхронное. Блокирующие операции (shell, файлы большого объёма) уходят в `asyncio.to_thread` или `asyncio.create_subprocess_exec`.

---

## 3. Структура репозитория

```
zero/
  pyproject.toml
  README.md
  ZERO_SPEC.md              # этот файл
  src/zero/
    __init__.py              # __version__, APP_NAME
    __main__.py              # `python -m zero` → запуск сервера
    cli.py                   # команда `zero`: serve, token, doctor, config path
    config.py                # загрузка/сохранение конфига, пути
    paths.py                 # пути macOS
    logging_setup.py
    server/
      app.py                 # FastAPI app factory
      ws.py                  # WebSocket-эндпоинт, роутинг сообщений
      auth.py                # локальный токен
      connections.py         # реестр подключённых клиентов
    protocol/
      messages.py            # pydantic-модели ВСЕХ сообщений протокола
      errors.py              # коды ошибок
    core/
      turn.py                # agent loop (один «ход»)
      model_call.py          # вызов модели: стрим ответа, ретраи, учёт каждого запроса
      context.py             # сборка промпта: system + память + история
      prompting.py           # промпт хода: память, история, саммари (этап 6)
      compaction.py          # сжатие старой истории в саммари (этап 6)
      expression.py          # позы/настроения
      cancellation.py
    providers/
      base.py                # LLMProvider, нормализованные типы сообщений/событий
      fake.py                # FakeProvider (скриптуемый)
      registry.py            # выбор провайдера по конфигу
    tools/
      base.py                # Tool, ToolResult, ToolContext, реестр
      builtin/
        files.py             # read_file, write_file, edit_file, list_dir, search_files
        shell.py             # run_command
        web.py               # web_fetch (web_search — позже, через провайдера поиска)
        memory.py            # memory_save, memory_search, memory_delete
        todo.py              # todo_* (этап 6)
        schedule.py          # schedule_* (этап 6)
        agents.py            # agent_* и ask_parent (этап 7)
      client_actions.py      # прокси для действий, объявленных клиентом
      permissions.py         # политики allow/ask/deny, sandbox путей
    approvals/
      manager.py             # ожидание ответа на подтверждение, таймауты
    store/
      db.py                  # подключение, миграции
      migrations/0001_init.sql ...
      sessions.py
      messages.py
      memory.py
      approvals.py           # сохранённые «always allow»
      todo.py, schedules.py, summaries.py   # этап 6
    scheduler/               # этап 6: rules.py (разбор «когда»), runner.py (фоновый запуск)
    gateways/telegram.py     # этап 9: Telegram-бот (телефон и уведомления)
    agents/                  # этап 7: runner.py (запуск, сообщения, остановка, доставка), types.py
  tests/
    conftest.py
    test_protocol.py
    test_turn_loop.py
    test_tools_files.py
    ...
  scripts/
    install_launchagent.py   # ставит/снимает LaunchAgent
```

---

## 4. Пути и конфиг (macOS)

```
~/Library/Application Support/Zero/
  config.json        # настройки (см. ниже)
  zero.db           # SQLite
  token              # локальный токен доступа, chmod 600
  files/             # вложения и файлы, отправленные пользователю
~/Library/Logs/Zero/zero.log   # ротация 5 × 5 MB
```

Каталог можно переопределить переменной `ZERO_HOME` (обязательно для тестов).

### `config.json`
```json
{
  "server": { "host": "127.0.0.1", "port": 47400 },
  "provider": { "name": "fake", "model": "fake-1", "options": {}, "history_token_budget": null },
  "agent": {
    "max_steps": 40,
    "max_steps_computer": 100,
    "turn_timeout_s": 900,
    "history_token_budget": 60000,
    "max_agents": 3
  },
  "permissions": {
    "bypass_approvals": false,
    "allowed_roots": ["~"],
    "denied_paths": ["~/.ssh", "~/Library/Keychains"],
    "tool_policies": {
      "read_file": "allow", "list_dir": "allow", "search_files": "allow",
      "write_file": "ask", "edit_file": "ask", "run_command": "ask",
      "web_fetch": "allow",
      "memory_save": "allow", "memory_search": "allow", "memory_delete": "ask"
    }
  },
  "secrets": {}
}
```
- Конфиг валидируется pydantic-моделью `Config`. Неизвестные ключи — предупреждение, не ошибка.
- **Секреты (API-ключи) в будущем хранятся в macOS Keychain** (`keyring`), а не в `config.json`. Пока провайдеров нет, это не реализуется. Поле `secrets` зарезервировано и должно оставаться пустым.

---

## 5. Запуск и жизненный цикл

- `python -m zero` (или `zero serve`) поднимает uvicorn на `127.0.0.1:<port>`.
- Если порт занят другим экземпляром Zero (проверка через `GET /health`), процесс завершается с кодом 0 и сообщением «уже запущен».
- При первом запуске генерируется `token` (32 байта, `secrets.token_urlsafe`) с правами `600`.
- Команда `zero`: `serve` (то же, что `python -m zero`), `token` (показать токен), `doctor` (папки, конфиг, версия схемы БД, права токена, порт, провайдер, `rg`; ничего не создаёт и не меняет, код выхода 1 при проблемах), `config path`, `usage` (запросы и токены по провайдерам и моделям: сегодня с 00:00 UTC и за 30 дней; база открывается только на чтение).
- Основной режим (этап 9): мозг запускает десктоп-приложение (живёт в строке меню, стартует при входе) как дочерний процесс `zero serve --parent-pid <pid>` и останавливает SIGTERM при выходе; мозг сам завершается, если процесс приложения исчез. LaunchAgent ниже — необязательный режим «всегда включено».
- `scripts/install_launchagent.py` ставит `~/Library/LaunchAgents/local.zero.brain.plist` и загружает его через `launchctl bootstrap gui/<uid>`: `RunAtLoad`, `KeepAlive: {SuccessfulExit: false}` (перезапуск только после падения, штатный выход «уже запущен» не зацикливается), `PATH` из окружения установки, вывод в `~/Library/Logs/Zero`. Флаги `--uninstall` и `--print`.
- Корректное завершение по SIGTERM/SIGINT (до того, как uvicorn рвёт соединения): новые `message.send` получают `error internal`, новые подключения закрываются с 1001, планировщик останавливается, все ходы отменяются и получают 5 с, чтобы закончиться штатно (потом обрываются). Ожидающие подтверждения отклоняются: клиенты получают `approval.resolved` с `decision: "deny"` и `by_client_id: "zero"`; ожидание `action.result` прекращается. Затем клиенты закрываются с кодом 1001, БД закрывается.

### HTTP-эндпоинты
| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | `{"app":"zero","version":"x.y.z","status":"ok"}`, без авторизации |
| GET | `/ws` | WebSocket, основной протокол |

Больше REST не нужно: всё идёт через WS.

---

## 6. Протокол (WebSocket, JSON)

### 6.1 Общее
- Один WS на клиента. Авторизация при подключении через заголовок `Authorization: Bearer <token>` или query `?token=` (для отладки). При неверном токене соединение закрывается с кодом `4401`.
- Каждое сообщение — JSON-объект с полем `type`. Сообщения, ожидающие ответа, несут `id` (UUID от отправителя), ответ ссылается на него через `reply_to`.
- Все типы описаны pydantic-моделями в `protocol/messages.py` (discriminated union по `type`). Неизвестный `type` даёт `error` с кодом `unknown_type`, соединение не рвётся.
- Версия протокола: `PROTOCOL_VERSION = 1`. Клиент передаёт её в `hello`. При несовпадении мажорной версии приходит `error` `protocol_mismatch` и закрытие.

### 6.2 Рукопожатие: клиент объявляет себя
Клиент → Zero:
```json
{
  "type": "hello",
  "id": "…",
  "protocol": 1,
  "client": {
    "app": "desktop",                       
    "client_id": "…",                         
    "display_name": "Desktop on MacBook",
    "character": {
      "name": "<имя персонажа>",
      "persona": "Markdown-текст персоны",
      "poses": ["idle", "sit", "wave"],
      "moods": ["neutral", "happy", "sad", "angry", "surprised"]
    },
    "environment": "macOS desktop app; replies show in a chat window that renders Markdown",
    "actions": [
      {
        "name": "screenshot",
        "description": "Capture the user's screen",
        "input_schema": { "type": "object", "properties": { "display": {"type":"integer"} } },
        "policy": "allow",
        "timeout_s": 15
      }
    ]
  }
}
```
Zero → клиент:
```json
{ "type": "hello.ok", "reply_to": "…", "zero_version": "0.1.0", "protocol": 1,
  "sessions": [ { "id": "…", "title": "…", "updated_at": "…" } ] }
```
- `actions` — **действия клиента**. Мозг превращает их в инструменты для модели (раздел 9.3). Если подключено несколько клиентов с одноимёнными действиями, действие вызывается у клиента, который начал текущий ход.
- `character` задаёт имя, персону и допустимые позы и настроения (раздел 10).
- Повторный `hello` в том же соединении обновляет объявление (например, сменили персонажа).

### 6.3 Клиент → Zero

| type | Поля | Ответ / эффект |
|---|---|---|
| `message.send` | `session_id?`, `text`, `attachments?: [{name, mime, data_b64 \| path}]`, `source: "desktop"\|"phone"` | `turn.started`, затем поток событий хода. Без `session_id` создаётся новая сессия. Вложения копируются в `files/` и хранятся как `ContentPart file` с путём (не base64); `path` проходит PathGuard, лимит 100 MB. Плохое вложение даёт `error invalid_message`, ход не начинается. В модель картинка уходит только из текущего сообщения (если модель видит картинки и картинка до 5 MB), остальные файлы — короткой текстовой заметкой с путём. |
| `turn.cancel` | `turn_id` | Ход отменяется (раздел 7.4), приходит `turn.finished` со `status: "cancelled"` |
| `approval.respond` | `approval_id`, `decision: "allow"\|"deny"\|"always"`, `note?` | Ход продолжается |
| `action.result` | `call_id`, `ok: bool`, `content: [ContentPart]`, `error?` | Результат действия клиента возвращается в agent loop |
| `session.list` | — | `session.list.ok` |
| `session.create` | `title?` | `session.created` |
| `session.rename` | `session_id`, `title` | `session.updated` |
| `session.delete` | `session_id` | `session.deleted` |
| `history.get` | `session_id`, `before?`, `limit=50`, `include_tools=true` | `history.ok` с сообщениями и вызовами инструментов |
| `ping` | — | `pong` |

### 6.4 Zero → клиент (события)
Все события хода несут `session_id` и `turn_id`. События рассылаются **всем** подключённым клиентам, подписанным на эту сессию (по умолчанию на все), чтобы десктоп и телефон видели одно и то же.

| type | Поля |
|---|---|
| `turn.started` | `turn_id`, `session_id`, `source` |
| `text.delta` | `delta` |
| `tool.started` | `call_id`, `tool`, `label` (короткое человекочитаемое: `$ ls -la`, `~/notes.md`), `input` |
| `tool.finished` | `call_id`, `status: "ok"\|"error"\|"denied"\|"cancelled"`, `summary` (`exit 1`, `+79 lines`, `3 matches`), `output_preview`, `duration_ms` |
| `action.call` | `call_id`, `action`, `input` — только клиенту, который должен исполнить |
| `approval.request` | `approval_id`, `tool`, `label`, `input`, `reason` (зачем), `diff?` (для правок файлов) |
| `approval.resolved` | `approval_id`, `decision`, `by_client_id` (чтобы другие клиенты убрали карточку). Если ход закончился раньше ответа (отмена, остановка сервиса), приходит `decision: "deny"`, `by_client_id: "zero"`. |
| `expression` | `pose`, `mood` — можно слать посреди хода |
| `file` | `call_id?`, `file: ContentPart file` — пользователю отдали файл (например, через `send_file`); клиент показывает его в чате |
| `turn.finished` | `status: "ok"\|"error"\|"cancelled"\|"max_steps"`, `text` (весь текст хода: склейка всех `text.delta` без разделителей), `expression: {pose, mood}`, `usage: {input_tokens, output_tokens}`, `error?` |
| `session.created` / `session.updated` / `session.deleted` | данные сессии: `{id, title, updated_at, kind}`, где `kind` — `chat` (разговор пользователя), `agent` или `scheduled`, чтобы клиент мог держать служебные сессии отдельно |
| `todo.updated` | `session_id`, `items: [{number, text, status}]` — весь список после изменения (этап 6) |
| `session.compacted` | `session_id`, `summary`, `until_message_id` — начало истории сжато в саммари (этап 6) |
| `agent.updated` | `agent: {id, name, parent_session_id, session_id?, task, status, report?, created_at, updated_at}` — агент запущен или сменил статус (этап 7) |

События хода агента (включая `approval.request` и `approval.resolved`) несут `agent: {id, name}`, чтобы клиент показал, кто действует и кто спрашивает. У обычных ходов этого поля нет.
| `error` | `code`, `message`, `reply_to?` |

### 6.5 ContentPart
Общий формат контента (результаты инструментов, вложения):
```json
{ "type": "text", "text": "…" }
{ "type": "image", "mime": "image/png", "data_b64": "…" }
{ "type": "file", "name": "report.pdf", "path": "…", "mime": "application/pdf" }   // mime необязателен
```

### 6.6 Коды ошибок
`unauthorized`, `protocol_mismatch`, `unknown_type`, `invalid_message`, `session_not_found`, `turn_busy` (в сессии уже идёт ход), `provider_error`, `provider_not_configured`, `tool_error`, `action_timeout`, `client_gone`, `internal`.

**Правило:** в одной сессии одновременно идёт не больше одного хода. Новое сообщение во время хода приходит `error turn_busy`. (Позже можно сделать очередь.)

---

## 7. Agent loop (`core/turn.py`)

### 7.1 Ход
```
on message.send:
  session = get_or_create
  persist user message
  emit turn.started
  messages = context.build(session)        # system + память + история + новое сообщение
  tools = registry.for_turn(client)        # встроенные + действия клиента
  for step in 1..max_steps:
      stream = provider.stream(messages, tools, model)
      for event in stream:
          TextDelta     → emit text.delta, копим текст
          ToolCall      → копим вызовы
          Usage         → копим usage
      if нет tool calls: break
      for call in tool_calls (последовательно в v1):
          result = await execute(call)       # с проверкой прав/подтверждением
          append assistant(tool_calls) + tool(result) в messages и в БД
      if в шаге есть текст и все вызовы — успешные set_expression: break
  persist assistant message (текст), если он не пустой или ход закончился с error/cancelled
  emit turn.finished
```
- `max_steps` берётся из конфига. Если в ходе использовались `computer_*`-действия, лимит равен `max_steps_computer`. Ход делает не больше `max_steps` вызовов модели. Если остался один вызов, а модель всё ещё просит инструменты, этот вызов идёт **без инструментов**, чтобы она подвела итог, и статус хода становится `max_steps`.
- Шаг, в котором модель уже написала текст, а все вызовы в нём — успешные `set_expression`, завершает ход без ещё одного вызова модели: ответ готов, а выражение продолжения не требует. Если текста в шаге ещё нет (модель сначала выбрала позу), цикл идёт дальше.
- Ошибка провайдера даёт до 2 ретраев с экспоненциальной задержкой (1 с, 4 с) на сетевых и 5xx-ошибках, затем `turn.finished status=error`. Ретрай делается, только пока в этом вызове клиентам не ушло ни одной `text.delta`: отправленный текст забрать нельзя, поэтому сбой посреди ответа сразу завершает ход с `error`, а частичный текст сохраняется.
- Пустой финальный ответ со статусом `ok` в историю не пишется: провайдеры не принимают пустые assistant-сообщения. Пустая запись остаётся только как метка хода, оборвавшегося с `error` или `cancelled`. Выражение отдельно не хранится: в истории оно есть как вызов `set_expression` с результатом, а клиенты получают его в событии `expression` и в `turn.finished`.
- Ошибка инструмента **не** ломает ход: она возвращается модели как `tool_result` с `is_error=true`.

### 7.2 Системный промпт (`core/context.py`)
Собирается из блоков в фиксированном порядке (стабильные блоки вверху, ради кэширования промпта у провайдеров):
1. Базовые правила мозга (кто ты, как пользоваться инструментами, как подтверждения работают).
2. Персона персонажа из `hello.character.persona`.
3. Окружение клиента из `hello.client.environment` и `source` сообщения (например, «пользователь пишет с телефона и не видит экран компьютера»).
4. Правила выражения: список поз и настроений, как их выбирать (раздел 10).
5. Память: топ-N фактов (раздел 11).
6. Текущие дата, время и часовой пояс (последним, потому что это меняется).

### 7.3 История и контекст
- История хранится полностью в БД, включая вызовы инструментов.
- В промпт уходят последние сообщения в пределах бюджета модели: `provider.history_token_budget`, а если он не задан — `agent.history_token_budget`. Токены считает `provider.count_tokens()` или эвристика: символы ASCII / 3,5 плюс остальные (кириллица и т. д.) / 2,5 — оценка нарочно с запасом.
- Изображения из старых ходов в промпт не идут, их заменяет `[image omitted]`. Исключение — текущий ход.
- Обрезка идёт по границам ходов: пара assistant `tool_use` + `tool_result` никогда не разрывается.
- (Этап 6) Когда история не влезает, окно сужается ещё на 1500 токенов (место под саммари), а всё, что до окна, сжимается в саммари сессии (`session_summaries`: текст и `until_message_id`). Саммари накопительное: в него добавляются только сообщения, выпавшие после прошлого раза. Пишет его модель (отдельный запрос, до 800 токенов, таймаут 60 с). С FakeProvider и при ошибке модели вместо этого берутся строки «User: …» по 200 символов. Длина саммари ограничена 4000 символов, при переполнении остаётся конец. Саммари идёт в системный промпт блоком `# Earlier in this conversation`, клиенты получают `session.compacted`.

### 7.4 Отмена
- У каждого хода есть `CancelToken`. `turn.cancel` его выставляет.
- Стриминг провайдера прерывается, `run_command` убивает процесс (SIGTERM, через 3 с SIGKILL), ожидание `action.result` и подтверждения прекращается.
- Частичный текст сохраняется в историю с пометкой `cancelled`.

---

## 8. Провайдеры (`providers/`)

### 8.1 Нормализованные типы
Внутри мозга используется **один** формат сообщений, независимый от провайдера:

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[ContentPart]            # text / image
    tool_calls: list[ToolCall] = []       # только assistant
    tool_call_id: str | None = None       # только tool
    is_error: bool = False                # только tool

class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict                    # JSON Schema

class ToolCall(BaseModel):
    id: str
    name: str
    input: dict
```

События стрима:
```python
TextDelta(text: str)
ToolCallEvent(call: ToolCall)            # провайдер сам собирает аргументы из дельт
UsageEvent(input_tokens: int, output_tokens: int)
StopEvent(reason: Literal["end", "tool_use", "max_tokens", "error"])
```

### 8.2 Интерфейс
```python
class LLMProvider(Protocol):
    name: str
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        *,
        max_tokens: int = 4096,
        cancel: CancelToken,
    ) -> AsyncIterator[StreamEvent]: ...

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec]) -> int | None: ...
    def supports_images(self, model: str) -> bool: ...
```
- Каждый адаптер сам переводит `Message` и `ToolSpec` в формат своего API и обратно.
- Ошибки адаптер переводит в `ProviderError(retryable: bool, message: str)`.
- `providers/registry.py` выдаёт провайдер по `config.provider.name`. Неизвестное имя даёт `provider_not_configured`.
- Каждый запрос к модели записывается в `model_calls`: провайдер, модель, назначение (`turn`, `agent`, `title`, `summary`), сессия, токены из `UsageEvent` и успех. Неудачные и отменённые запросы тоже пишутся: бесплатные тарифы ограничивают и число запросов. Ошибка записи не ломает ответ.

### 8.3 FakeProvider
Нужен для разработки и **всех** тестов.
- Принимает скрипт: список шагов, каждый шаг — список событий (`TextDelta`, `ToolCallEvent`, …).
- Режим `echo` по умолчанию отвечает «echo: <текст пользователя>» по кускам и вызывает `set_expression`.
- Сохраняет все полученные `messages` и `tools`, чтобы тесты могли проверить, что ушло в модель.
- Может имитировать задержку, ошибку (`retryable` true/false) и бесконечный стрим (для теста отмены).

---

## 9. Инструменты (`tools/`)

### 9.1 Базовый интерфейс
```python
class Tool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]          # → JSON Schema для модели
    default_policy: Literal["allow", "ask", "deny"]
    def label(self, input) -> str: ...    # короткая строка для UI
    async def run(self, input, ctx: ToolContext) -> ToolResult: ...

class ToolContext:
    session_id: str; turn_id: str; client_id: str
    cancel: CancelToken
    config: Config
    paths: PathGuard

class ToolResult:
    content: list[ContentPart]
    is_error: bool = False
    summary: str = ""                     # "exit 0", "+12 -3", "5 matches"
```
- Выход инструмента обрезается до 30 000 символов: голова и хвост, посередине `[… N chars truncated …]`.
- Ввод валидируется pydantic. При невалидном вводе модель получает `is_error=true` с понятным текстом ошибки и шанс исправиться.

### 9.2 Встроенные инструменты v1

| Инструмент | Политика | Что делает |
|---|---|---|
| `read_file(path, offset?, limit?)` | allow | Текстовый файл с номерами строк. Изображения отдаёт как image. Бинарные файлы — ошибка. |
| `list_dir(path, depth=1)` | allow | Дерево с размерами, скрытые файлы по флагу |
| `search_files(pattern, path, glob?)` | allow | Регекс-поиск (через `rg`, если есть, иначе Python) |
| `write_file(path, content)` | ask | Создать или перезаписать. В подтверждении показывается дифф. |
| `edit_file(path, old, new)` | ask | Точная замена, `old` должен встречаться ровно 1 раз |
| `run_command(command, cwd?, timeout_s=120)` | ask | `zsh -lc` в своей группе процессов, stdout+stderr, код выхода. Таймаут и отмена: SIGTERM группе, через 3 с SIGKILL. Фоновые задачи команды останавливаются, когда она заканчивается, поэтому серверы так не запускают. |
| `web_fetch(url)` | allow | Скачать страницу и извлечь основной текст (trafilatura). Только http(s), до 5 MB, таймаут 20 с, до 5 редиректов. Адрес этого компьютера или локальной сети требует подтверждения (даже при bypass); редирект на такой адрес с другого хоста отклоняется. |
| `memory_save(text, tags?)` | allow | Раздел 11 |
| `memory_search(query)` | allow | Раздел 11 |
| `memory_delete(id)` | ask | Раздел 11 |
| `set_expression(pose, mood)` | allow | Раздел 10 |
| `send_file(path)` | allow | Отдать файл пользователю: копия в `files/<id>/`, событие `file`. До 100 MB. |
| `todo_add(text)`, `todo_update(number, status)`, `todo_list()` | allow | Список дел сессии, пункты нумеруются 1, 2, 3…; статусы `pending`/`in_progress`/`done`. После изменения — `todo.updated`. |
| `schedule_create(task, when, timezone?)` | allow | `when`: `in 2m`, `at 2026-11-12T15:30`, `every 30m` (не чаще раза в минуту), `daily at 09:00`, `every mon,wed at 09:00`, `every weekdays at 08:30`. Пояс по умолчанию — пояс компьютера; недельные правила держат местное время при переходе на летнее. |
| `schedule_list()` | allow | Расписания с ближайшим запуском |
| `schedule_delete(id)` | ask | Удалить расписание |
| `agent_start(name, task)` | allow | Запустить агента (этап 7). Агент видит только `task`. |
| `agent_message(id, text)` | allow | Ответ, поправка или новая работа агенту своей сессии; занятому агенту — ошибка |
| `agent_status(id)` | allow | Статус, число шагов, последний вывод, отчёт |
| `agent_stop(id)` | allow | Отменить работу и удалить сессию агента; запись и отчёт остаются |
| `ask_parent(question)` | allow | Только у агентов: вопрос сессии, которая их запустила |

### 9.3 Действия клиента
- Каждое `action` из `hello` превращается в инструмент с тем же именем, описанием и схемой.
- Вызов: мозг шлёт `action.call` нужному клиенту и ждёт `action.result` с таким же `call_id` не дольше `timeout_s`.
- Если клиент отключился, вызов возвращает `client_gone`. По таймауту возвращается `action_timeout`. Обе ошибки уходят модели как tool error.
- `call_id` в `action.call` выдаёт мозг (UUID), он не совпадает с id вызова от модели. Ответить на вызов может только то подключение, которому он отправлен; чужой или повторный `action.result` игнорируется.
- Действие не может подменить встроенный инструмент или `set_expression`: при совпадении имён действие пропускается.
- Политика берётся из объявления клиента, но **конфиг мозга может её только ужесточить** (`allow → ask → deny`), но не ослабить.
- Типовые действия десктоп-клиента (для справки, реализуются в «теле»): `screenshot`, `computer_view`, `computer_act`, `open_url`.

### 9.4 Права и песочница (`tools/permissions.py`)
- **PathGuard:** любой путь раскрывается (`~`, симлинки через `resolve()`) и должен лежать внутри одного из `allowed_roots` и вне всех `denied_paths`. Иначе ошибка без подтверждения.
- Порядок решения: `deny` в конфиге → deny. Сохранённое «always allow» для этого инструмента → allow. `bypass_approvals=true` → allow. Иначе применяется политика инструмента.
- **`bypass_approvals` никогда не отменяет `deny`.**
- Пароли и секреты: `read_file` отказывает для путей из `denied_paths` и для файлов вида `*.pem`, `id_*`, `.env*` (с политикой `ask` даже при bypass).

---

## 10. Выражение персонажа (`core/expression.py`)

- Модель выбирает позу и настроение через инструмент `set_expression(pose, mood)`. В схеме `enum` строится из `hello.character.poses` и `moods`.
- Инструмент можно вызвать в любой момент хода. Каждый вызов шлёт событие `expression`.
- В `turn.finished.expression` попадает последнее выбранное выражение, а если модель его не выбрала — `{pose: poses[0], mood: "neutral" или moods[0]}`.
- `set_expression` дёшев: он не показывается как отдельная строка инструмента в UI (у `tool.started` выставлен флаг `hidden: true`).
- В системном промпте стоит правило: выражение вместо эмодзи и действий в звёздочках.

---

## 11. Память (`store/memory.py`, `tools/builtin/memory.py`)

v1 устроена просто и предсказуемо:
- Таблица `memories(id, text, tags, created_at, updated_at, uses)`.
- `memory_search` работает через **SQLite FTS5** по `text` и `tags`. Векторы не используются (их можно добавить позже за тем же интерфейсом `MemoryStore.search()`).
- Запрос разбивается на слова, каждое берётся в кавычки (синтаксис FTS из запроса безопасен) и ищется по префиксу грубой основы: у слов от 5 букв отбрасываются 2 последние, но остаётся не меньше 4 («Прага» находит «Праге»). Слова соединяются через `OR`, порядок — по bm25.
- Тот же текст при повторном сохранении не создаёт дубль, а обновляет запись.
- В системный промпт попадают **10** записей: половина — чаще всего используемые (`uses > 0`), остальное — самые свежие, так что новый факт виден сразу. Плюс подсказка, что остальное ищется через `memory_search`.
- Модель сохраняет в память только то, что пользователь сказал о себе или явно попросил запомнить. Это правило стоит в базовом промпте.

---

## 12. Хранилище (SQLite)

`store/migrations/0001_init.sql`:
```sql
CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  title TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  turn_id TEXT,
  role TEXT NOT NULL,              -- user | assistant | tool
  content_json TEXT NOT NULL,      -- list[ContentPart]; картинки как путь в files/, не base64
  tool_calls_json TEXT,
  tool_call_id TEXT,
  is_error INTEGER NOT NULL DEFAULT 0,
  status TEXT,                     -- ok | cancelled | error
  source TEXT,                     -- desktop | phone | scheduler | agent
  created_at TEXT NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);

CREATE TABLE tool_runs (            -- версия 2: свой id, call_id может повторяться
  id TEXT PRIMARY KEY,
  call_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  tool TEXT NOT NULL,
  label TEXT,
  input_json TEXT,
  status TEXT,
  summary TEXT,
  output_preview TEXT,
  duration_ms INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_tool_runs_turn ON tool_runs(session_id, turn_id);

CREATE TABLE memories (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, tags TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, uses INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE memories_fts USING fts5(text, tags, content='memories', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2');   -- миграция 0003, плюс триггеры синхронизации

CREATE TABLE approval_rules (
  tool TEXT PRIMARY KEY, decision TEXT NOT NULL, created_at TEXT NOT NULL
);

-- этап 6: 0004_todo, 0005_schedules, 0006_summaries
CREATE TABLE todos (id, session_id, number, text, status, created_at, updated_at,
  UNIQUE (session_id, number));
CREATE TABLE schedules (id, task, rule_json, rule_text, timezone, next_run_at, last_run_at,
  enabled, created_at);            -- next_run_at NULL у отработавшего разового
CREATE TABLE session_summaries (session_id PRIMARY KEY, text, until_message_id, updated_at);

-- 0009_session_kinds: sessions.kind TEXT NOT NULL DEFAULT 'chat' — chat | agent | scheduled

-- 0008_model_calls: каждый запрос к модели (без внешнего ключа: учёт переживает сессии)
CREATE TABLE model_calls (id, provider, model, purpose, session_id, input_tokens,
  output_tokens, ok, created_at);

-- этап 7: 0007_agents
CREATE TABLE agents (id, name, parent_session_id REFERENCES sessions ON DELETE CASCADE,
  session_id UNIQUE REFERENCES sessions ON DELETE SET NULL, task, status, report,
  created_at, updated_at);   -- status: working | asking | waiting_user | done | failed | stopped
```
- Время хранится в ISO-8601 UTC. Id — UUID4 (или UUIDv7 для сортируемости).
- Схема меняется только новыми файлами миграций. `0002_tool_runs_own_id.sql` дала `tool_runs` собственный `id`: `call_id` придумывает модель, и он может повторяться между ходами.
- Заголовок сессии: после первого хода (с любым статусом) мозг делает короткий отдельный запрос к модели (5–7 слов) и шлёт `session.updated` уже после `turn.finished`. С FakeProvider запроса нет: берутся первые 6 слов сообщения. Они же служат запасным вариантом, если модель упала, не ответила за 15 с или вернула пустой ответ. Название, которое дал пользователь, автозаголовок не затирает, `updated_at` при этом не меняется.

---

## 13. Безопасность

- Сервер слушает **только** `127.0.0.1`. Запуск на `0.0.0.0` запрещён конфигом-валидатором.
- Локальный токен обязателен для WS. Файл `token` имеет права `600`. Токен не пишется в логи.
- Проверка `Origin` у WS: разрешены отсутствие заголовка (нативные клиенты) и `null`, браузерные origin отклоняются. Это защищает от атаки «вредный сайт стучится на localhost».
- Подтверждения неотключаемы для `deny` и для путей из `denied_paths`.
- Логи не содержат содержимого файлов, ответов модели целиком и секретов. В них идут метаданные: инструмент, длительность, статус.
- Телефонный доступ — Telegram-бот, модуль мозга (этап 9). Бот сам забирает сообщения (long polling, только исходящие запросы): входящих портов нет, этот WS наружу **не** открывается. Принимаются сообщения только из `telegram.allowed_chat_id`, остальные игнорируются и не логируются. Токен бота хранится в Keychain.

---

## 14. Этапы и критерии приёмки

### Этап 1 — каркас
- Структура репо, `pyproject.toml` (uv), ruff, pytest, `python -m zero` поднимает сервер.
- `paths.py`, `config.py` (pydantic, `ZERO_HOME`), логирование с ротацией, генерация токена.
- `GET /health`.
- **Готово, когда:** сервер стартует, `/health` отвечает, второй запуск видит первый и выходит, тесты конфига и путей зелёные.

### Этап 2 — протокол и сессии
- Все pydantic-модели из раздела 6. WS с авторизацией и Origin-проверкой, `hello`, `ping`, сессии CRUD, `history.get`.
- SQLite и миграции.
- **Готово, когда:** тест через `TestClient` проходит hello → создание сессии → переименование → список → удаление. Неверный токен даёт 4401, неизвестный type даёт `error`.

### Этап 3 — agent loop + FakeProvider
- `providers/base.py`, `fake.py`, `registry.py`. `core/turn.py`, `context.py`, `expression.py`, `cancellation.py`.
- `message.send` → поток `turn.started` / `text.delta` / `expression` / `turn.finished`. История сохраняется.
- **Готово, когда:** тесты покрывают: echo-ход, ход с инструментом `set_expression`, отмену посреди стрима, ретрай retryable-ошибки, `turn_busy`, лимит шагов с финальным шагом без инструментов, сборку системного промпта (порядок блоков, обрезку истории без разрыва пар tool_use/tool_result).

### Этап 4 — инструменты и подтверждения
- `tools/base.py`, реестр, PathGuard, политики, `approvals/manager.py`, «always allow».
- Встроенные: files, shell, web_fetch, send_file.
- **Готово, когда:** для каждого инструмента есть тест (в `tmp_path`). Подтверждение allow / deny / always проверено. Путь вне `allowed_roots` отклоняется без запроса. bypass не пробивает deny. Отмена убивает `run_command`. Дифф попадает в `approval.request`.

### Этап 5 — действия клиента + память
- `client_actions.py`: action → tool, `action.call` / `action.result`, таймаут, `client_gone`, ужесточение политики конфигом.
- Память на FTS5 и её инструменты. Память в системном промпте.
- Рассылка событий нескольким клиентам и `approval.resolved`.
- **Готово, когда:** тест с двумя WS-клиентами: один объявляет `screenshot`, FakeProvider его вызывает, результат-картинка доходит до модели (видно в записанных `messages`). Второй клиент видит те же события. Подтверждение, отвеченное с одного клиента, закрывается у другого.

### Этап 6 — расписания, todo, длинный контекст
- `todo_*`, `schedule_*` (once / cron-like), планировщик запускает ход в выделенной сессии с `source: "scheduler"` и шлёт события.
- Планировщик проверяет расписания раз в 30 с. Наступившее расписание сначала получает новый `next_run` (разовое выключается), потом запускается ход в сессии `scheduled-<id>` (создаётся при первом запуске, `session.created` всем). Пока прошлый запуск идёт, новый пропускается.
- Сжатие старой истории в саммари.
- LaunchAgent-скрипт, команда `zero`, корректное завершение.
- Номера миграций сдвинуты на 1 относительно роадмапа: `0003` занята памятью.

### Этап 7 — агенты
- `agent_start(task)`, `agent_message`, `agent_status`, `agent_stop`. Агент — это отдельная сессия с тем же набором инструментов, собственным контекстом (только задача) и своим ходом в фоне. Его подтверждения идут пользователю с пометкой имени агента. Вопросы агента идут родительской модели.
- Сессия агента называется `Agent: <name>`. В системном промпте блок `# Your role` после окружения; ход идёт с `source: "agent"`. Инструменты — все, кроме `agent_*`, плюс `ask_parent`: агенты не запускают агентов (это проверяется и в раннере).
- Статусы: `working` — ход идёт; `waiting_user` — ход ждёт подтверждения пользователя; `asking` — агент задал вопрос и ждёт ответа; `done` (отчёт — последний текст хода), `failed` (отчёт — ошибка), `stopped`. Управлять можно только агентами своей сессии.
- Всё, что агент говорит родителю, приходит в родительскую сессию сообщением пользователя с `source: "agent"` и запускает там ход, как только сессия свободна: «Agent '<name>' (id …) asks: …», «… has finished. Its report: …», «… failed: …». Вопросы уходят после конца хода агента, чтобы он был свободен принять ответ. Пока сервис останавливается, такие ходы не начинаются.
- Одновременно не больше `agent.max_agents` (по умолчанию 3) незавершённых агентов (`working`, `asking`, `waiting_user`). После перезапуска незавершённые агенты помечаются `failed`.

### Этап 8 — реальные провайдеры (по команде пользователя)
- Адаптеры поверх `LLMProvider`, ключи в Keychain, выбор провайдера и модели в конфиге.

### Этап 9 — уведомления и Telegram (после этапа 8)
- Событие `notification` и инструмент `notify_user(text)`; ходы планировщика и отчёты агентов сообщают о себе уведомлением.
- `zero serve --parent-pid`: мозг живёт вместе с приложением.
- Telegram-бот: переписка с Зеро (`source: "phone"`), подтверждения кнопками, уведомления по правилу `notifications.telegram` (`away` по умолчанию: когда клиенты сообщили `presence.update {away: true}` или их нет).

---

## 15. Соглашения по коду

- Модули маленькие (≤ 400 строк). Одна ответственность на модуль. Исключение — `protocol/messages.py`: в нём только объявления моделей протокола, и он может быть длиннее.
- Никаких глобальных синглтонов. Зависимости собираются в `server/app.py` и передаются явно (`AppState`).
- Публичные функции и классы снабжаются короткими docstring.
- Сообщения коммитов — короткая фраза на английском в повелительном наклонении, без префиксов вида `feat:` или `chore:`.
- Логи через `logging.getLogger(__name__)`. `print` запрещён.
- Время только через `datetime.now(UTC)`, обёрнутое в `clock.now()`, чтобы его можно было подменить в тестах.

---

## 16. Открытые вопросы (решить позже)
1. Финальное имя проекта, персонажа и bundle id.
2. Какие провайдеры и модели будут первыми.
3. Нужна ли очередь сообщений вместо `turn_busy`.
4. ~~Где живёт телефонный шлюз~~ — решено: Telegram-бот, модуль мозга (этап 9).
5. Векторная память: нужна ли и на чём (sqlite-vec, отдельный сервис).
6. MCP-клиент в мозге (этап 7+).

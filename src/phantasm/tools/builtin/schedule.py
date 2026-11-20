"""Built-in schedule tools: ``schedule_create``, ``schedule_list`` and ``schedule_delete``."""

from typing import Any

from pydantic import BaseModel, Field

from phantasm import clock
from phantasm.scheduler.rules import HELP, RuleError, describe, local_zone, parse_rule, zone_named
from phantasm.store.schedules import Schedule, SchedulesRepo
from phantasm.tools.base import Preview, Tool, ToolContext, ToolResult, error_result, text_result

MAX_TASK_CHARS = 1000
LABEL_CHARS = 60


def _short(text: str) -> str:
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 1] + "…"


def _line(schedule: Schedule) -> str:
    zone = zone_named(schedule.timezone)
    when = describe(schedule.rule, zone)
    upcoming = (
        f"next {schedule.next_run_at.astimezone(zone):%a %Y-%m-%d %H:%M} ({schedule.timezone})"
        if schedule.next_run_at
        else "finished"
    )
    return f"- [{schedule.id}] {when}, {upcoming}: {schedule.task}"


class ScheduleCreateInput(BaseModel):
    task: str = Field(
        min_length=1,
        max_length=MAX_TASK_CHARS,
        description="What to do when it is time, written as a request to yourself",
    )
    when: str = Field(description=HELP)
    timezone: str | None = Field(
        default=None, description="IANA time zone, e.g. Europe/Prague; the computer's by default"
    )


class ScheduleCreate(Tool[ScheduleCreateInput]):
    """Plans a task for later, once or again and again."""

    name = "schedule_create"

    core = False
    description = (
        "Plan a task to run later on your own, once or repeatedly, e.g. a reminder. When it "
        "is time, you get the task as a message in a separate conversation. " + HELP
    )
    input_model = ScheduleCreateInput
    default_policy = "allow"

    def __init__(self, schedules: SchedulesRepo) -> None:
        self._schedules = schedules

    def label(self, params: ScheduleCreateInput) -> str:
        return f"{params.when}: {_short(params.task)}"

    async def run(self, params: ScheduleCreateInput, ctx: ToolContext) -> ToolResult:
        try:
            zone = zone_named(params.timezone) if params.timezone else local_zone()
            rule = parse_rule(params.when, clock.now(), zone)
            schedule = await self._schedules.add(params.task, rule, params.when, zone.key)
        except (RuleError, ValueError) as exc:
            return error_result(f"Nothing was scheduled: {exc}")
        return text_result(f"Scheduled: {_line(schedule)[2:]}", summary=describe(rule, zone))


class ScheduleListInput(BaseModel):
    pass


class ScheduleList(Tool[ScheduleListInput]):
    """Shows the planned tasks."""

    name = "schedule_list"

    core = False
    description = "List the planned tasks with their id, when they run next and what they do."
    input_model = ScheduleListInput
    default_policy = "allow"

    def __init__(self, schedules: SchedulesRepo) -> None:
        self._schedules = schedules

    async def run(self, params: ScheduleListInput, ctx: ToolContext) -> ToolResult:
        schedules = await self._schedules.list()
        if not schedules:
            return text_result("Nothing is scheduled.", summary="0 schedules")
        count = len(schedules)
        return text_result(
            "\n".join(_line(schedule) for schedule in schedules),
            summary=f"{count} schedule{'' if count == 1 else 's'}",
        )


class ScheduleDeleteInput(BaseModel):
    id: str = Field(min_length=1, description="Id of the schedule, as schedule_list shows it")


class ScheduleDelete(Tool[ScheduleDeleteInput]):
    """Cancels a planned task."""

    name = "schedule_delete"

    core = False
    description = "Cancel a planned task by its id. The user usually approves this first."
    input_model = ScheduleDeleteInput
    default_policy = "ask"

    def __init__(self, schedules: SchedulesRepo) -> None:
        self._schedules = schedules

    def label(self, params: ScheduleDeleteInput) -> str:
        return f"cancel {params.id}"

    async def preview(self, params: ScheduleDeleteInput, ctx: ToolContext) -> Preview:
        schedule = await self._schedules.get(params.id)
        if schedule is None:
            return Preview(refusal=f"There is no schedule with id {params.id}.")
        return Preview(label=f"cancel: {_short(schedule.task)}")

    async def run(self, params: ScheduleDeleteInput, ctx: ToolContext) -> ToolResult:
        schedule = await self._schedules.get(params.id)
        if schedule is None or not await self._schedules.delete(params.id):
            return error_result(f"There is no schedule with id {params.id}.")
        return text_result(f"Cancelled: {schedule.task}", summary="cancelled")


def schedule_tools(schedules: SchedulesRepo) -> list[Tool[Any]]:
    """The three schedule tools, working on ``schedules``."""
    return [ScheduleCreate(schedules), ScheduleList(schedules), ScheduleDelete(schedules)]

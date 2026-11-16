"""Background jobs: a ``run_command`` that runs longer than a moment goes on as one.

A job keeps running in its own process group after the turn moves on, and all
of its output goes to a log file in ``outputs/jobs``. ``job_output`` gives what
is new since the last look (and can wait for the end), ``job_stop`` stops it.
A job still ends at its command's timeout, cancelling the turn leaves it alone,
and every job stops with the brain.
"""

import asyncio
import contextlib
import signal
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel, Field

from zero.paths import outputs_dir
from zero.protocol.messages import TextPart
from zero.tools.base import Tool, ToolContext, ToolResult, error_result
from zero.tools.builtin import shell

MAX_RUNNING_JOBS = 4
MAX_WAIT_S = 60

Outcome = Literal["running", "exit", "timeout", "stopped"]


class Job:
    """One command running in the background."""

    def __init__(
        self,
        job_id: str,
        label: str,
        process: asyncio.subprocess.Process,
        reading: asyncio.Task[None],
        exiting: asyncio.Task[None],
        output: "shell.Output",
        log: BinaryIO,
        log_path: Path,
        timeout_s: int,
    ) -> None:
        self.id = job_id
        self.label = label
        self.process = process
        self.reading = reading
        self.exiting = exiting
        self.output = output
        self.log = log
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.started = asyncio.get_running_loop().time()
        self.outcome: Outcome = "running"
        self.read_offset = log.tell()
        self.stop_requested = asyncio.Event()
        self.done = asyncio.Event()
        self.watcher: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self.outcome == "running"

    def status(self) -> tuple[str, str, bool]:
        """The status line, the summary for the UI and whether it counts as an error."""
        if self.outcome == "running":
            seconds = int(asyncio.get_running_loop().time() - self.started)
            return f"running for {seconds} s", "running", False
        if self.outcome == "timeout":
            return f"stopped after the {self.timeout_s} s timeout", "timed out", True
        if self.outcome == "stopped":
            return "stopped", "stopped", False
        code = self.process.returncode
        assert code is not None
        if code < 0:
            name = signal.Signals(-code).name if -code in signal.valid_signals() else str(-code)
            return f"killed by {name}", f"killed by {name}", True
        return f"exit {code}", f"exit {code}", code != 0

    def read_new(self) -> str:
        """Output written since the last read."""
        try:
            with self.log_path.open("rb") as file:
                file.seek(self.read_offset)
                data = file.read()
        except OSError:
            return ""
        self.read_offset += len(data)
        return data.decode(errors="replace")


class JobManager:
    """The brain's background jobs, named job-1, job-2 and so on."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._count = 0

    def running(self) -> int:
        return sum(1 for job in self._jobs.values() if job.running)

    def has_room(self) -> bool:
        return self.running() < MAX_RUNNING_JOBS

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def describe_all(self) -> str:
        if not self._jobs:
            return "There are no jobs."
        jobs = ", ".join(
            f"{job.id} ({job.label}, {job.status()[0]})" for job in self._jobs.values()
        )
        return f"Jobs: {jobs}."

    def adopt(
        self,
        label: str,
        process: asyncio.subprocess.Process,
        reading: asyncio.Task[None],
        exiting: asyncio.Task[None],
        output: "shell.Output",
        remaining_s: float,
        timeout_s: int,
    ) -> Job:
        """Take over a running command: its output so far and from now on goes to a log."""
        self._count += 1
        job_id = f"job-{self._count}"
        folder = outputs_dir() / "jobs"
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"{job_id}.log"
        log = log_path.open("wb")
        log.write(output.text().encode())
        log.flush()
        output.sink = log
        job = Job(job_id, label, process, reading, exiting, output, log, log_path, timeout_s)
        self._jobs[job_id] = job
        job.watcher = asyncio.create_task(self._watch(job, remaining_s))
        return job

    async def _watch(self, job: Job, remaining_s: float) -> None:
        outcome: Outcome = "stopped"
        stop = asyncio.create_task(job.stop_requested.wait())
        try:
            await asyncio.wait(
                {job.exiting, stop}, timeout=remaining_s, return_when=asyncio.FIRST_COMPLETED
            )
            if job.exiting.done():
                outcome = "exit"
            elif not job.stop_requested.is_set():
                outcome = "timeout"
            await shell.stop_group(job.process, job.reading)
            await asyncio.wait({job.reading}, timeout=shell.DRAIN_S)
        finally:
            stop.cancel()
            job.reading.cancel()
            job.exiting.cancel()
            if job.process.returncode is None:
                shell.signal_group(job.process, signal.SIGKILL)
                await shell.exited(job.process)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.process.wait(), shell.DRAIN_S)
            job.output.sink = None
            with contextlib.suppress(OSError):
                job.log.close()
            job.outcome = outcome
            job.done.set()

    async def stop(self, job: Job) -> None:
        job.stop_requested.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(job.done.wait(), shell.STOP_GRACE_S + 2 * shell.DRAIN_S + 1)

    async def stop_all(self) -> None:
        """Stop every job that still runs; the brain is shutting down."""
        await asyncio.gather(*(self.stop(job) for job in self._jobs.values() if job.running))


def _report(job: Job) -> ToolResult:
    new = job.read_new().rstrip("\n") or "(no new output)"
    line, summary, is_error = job.status()
    return ToolResult(
        content=[TextPart(text=f"{new}\n[{job.id}: {line}]")], is_error=is_error, summary=summary
    )


class JobOutputInput(BaseModel):
    job_id: str = Field(description="The job, such as job-1")
    wait_s: int = Field(
        default=0,
        ge=0,
        le=MAX_WAIT_S,
        description="Seconds to wait for the job to end before answering; 0 answers at once",
    )


class JobOutput(Tool[JobOutputInput]):
    """Reads what a background job printed since the last look."""

    name = "job_output"

    core = False
    description = (
        "Get what a background job of run_command printed since you last looked, and "
        "whether it still runs or how it ended. wait_s waits up to that many seconds for "
        "it to end first."
    )
    input_model = JobOutputInput
    default_policy = "allow"

    def __init__(self, jobs: JobManager) -> None:
        self._jobs = jobs

    def label(self, params: JobOutputInput) -> str:
        return params.job_id

    async def run(self, params: JobOutputInput, ctx: ToolContext) -> ToolResult:
        job = self._jobs.get(params.job_id)
        if job is None:
            return error_result(f"There is no job {params.job_id}. {self._jobs.describe_all()}")
        if params.wait_s and job.running:
            done = asyncio.create_task(job.done.wait())
            cancelled = asyncio.create_task(ctx.cancel.wait())
            try:
                await asyncio.wait(
                    {done, cancelled}, timeout=params.wait_s, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                done.cancel()
                cancelled.cancel()
        return _report(job)


class JobStopInput(BaseModel):
    job_id: str = Field(description="The job, such as job-1")


class JobStop(Tool[JobStopInput]):
    """Stops a background job and everything it started."""

    name = "job_stop"

    core = False
    description = (
        "Stop a background job of run_command and everything it started; you get the "
        "output it printed since you last looked."
    )
    input_model = JobStopInput
    default_policy = "allow"

    def __init__(self, jobs: JobManager) -> None:
        self._jobs = jobs

    def label(self, params: JobStopInput) -> str:
        return params.job_id

    async def run(self, params: JobStopInput, ctx: ToolContext) -> ToolResult:
        job = self._jobs.get(params.job_id)
        if job is None:
            return error_result(f"There is no job {params.job_id}. {self._jobs.describe_all()}")
        if job.running:
            await self._jobs.stop(job)
        return _report(job)


def job_tools(jobs: JobManager) -> list[Tool[BaseModel]]:
    return [JobOutput(jobs), JobStop(jobs)]  # type: ignore[list-item]

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class LaneWorkItem:
    """Model-independent payload claimed by one singleton lane."""

    request_id: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("lane work item request_id must not be empty")


@dataclass(frozen=True, slots=True)
class LaneWorkResult:
    """Terminal result returned before a lane asks for more work."""

    request_id: str
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("lane work result request_id must not be empty")


class SingletonLaneWorkerPool:
    """Asynchronous producer-consumer pool for the width-one EPAC limit.

    Rank 0 owns admission and completion state. Every rank, including rank 0,
    independently claims and executes one request at a time. Remote workers use
    pre-created CPU control groups, so no WORLD collective appears between jobs.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        control_group: Callable[[int], object],
        claim: Callable[[int], LaneWorkItem | None],
        execute: Callable[[LaneWorkItem, int], LaneWorkResult],
        complete: Callable[[LaneWorkResult], None],
        should_stop: Callable[[], bool],
        idle_sleep_s: float = 0.01,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid singleton lane worker rank/world_size")
        if idle_sleep_s <= 0:
            raise ValueError("idle_sleep_s must be positive")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._control_group = control_group
        self._claim = claim
        self._execute = execute
        self._complete = complete
        self._should_stop = should_stop
        self._idle_sleep_s = float(idle_sleep_s)
        self._shutdown = threading.Event()
        self._errors: queue.Queue[BaseException] = queue.Queue()

    def run(self) -> None:
        if self.rank == 0:
            self._run_coordinator_and_local_lane()
        else:
            self._run_remote_lane()

    def _stopping(self) -> bool:
        return self._shutdown.is_set() or self._should_stop()

    def _execute_safely(self, item: LaneWorkItem, rank: int) -> LaneWorkResult:
        try:
            return self._execute(item, rank)
        except BaseException as exc:
            return LaneWorkResult(request_id=item.request_id, error=str(exc))

    def _run_coordinator_and_local_lane(self) -> None:
        listeners = [
            threading.Thread(
                target=self._serve_remote_lane,
                args=(peer,),
                name=f"epac-lane-control-{peer}",
                daemon=True,
            )
            for peer in range(1, self.world_size)
        ]
        for thread in listeners:
            thread.start()
        try:
            while not self._stopping():
                self._raise_listener_error()
                item = self._claim(0)
                if item is None:
                    time.sleep(self._idle_sleep_s)
                    continue
                self._complete(self._execute_safely(item, 0))
        finally:
            self._shutdown.set()
            for thread in listeners:
                thread.join()
            self._raise_listener_error()

    def _serve_remote_lane(self, peer: int) -> None:
        group = self._control_group(peer)
        try:
            while True:
                message = [None]
                dist.recv_object_list(
                    message,
                    src=peer,
                    group=group,
                    device=torch.device("cpu"),
                )
                request = message[0] or {}
                result = request.get("result")
                if result is not None:
                    if not isinstance(result, LaneWorkResult):
                        raise TypeError("remote lane returned an invalid result")
                    self._complete(result)
                if request.get("stopping") or self._stopping():
                    reply = {"action": "stop"}
                    dist.send_object_list(
                        [reply],
                        dst=peer,
                        group=group,
                        device=torch.device("cpu"),
                    )
                    return
                item = self._claim(peer)
                reply = (
                    {"action": "idle"}
                    if item is None
                    else {"action": "run", "item": item}
                )
                dist.send_object_list(
                    [reply],
                    dst=peer,
                    group=group,
                    device=torch.device("cpu"),
                )
        except BaseException as exc:
            self._errors.put(exc)
            self._shutdown.set()

    def _run_remote_lane(self) -> None:
        group = self._control_group(self.rank)
        result: LaneWorkResult | None = None
        while True:
            dist.send_object_list(
                [{"result": result, "stopping": self._stopping()}],
                dst=0,
                group=group,
                device=torch.device("cpu"),
            )
            result = None
            reply = [None]
            dist.recv_object_list(
                reply,
                src=0,
                group=group,
                device=torch.device("cpu"),
            )
            command = reply[0] or {}
            action = command.get("action")
            if action == "stop":
                return
            if action == "idle":
                time.sleep(self._idle_sleep_s)
                continue
            if action != "run" or not isinstance(command.get("item"), LaneWorkItem):
                raise RuntimeError("singleton lane received an invalid command")
            item = command["item"]
            result = self._execute_safely(item, self.rank)

    def _raise_listener_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("singleton lane control thread failed") from error


class FullWorldLaneWorkerPool:
    """Producer-consumer loop for the full-world CP topology limit.

    The control collective is scoped to the lane, which happens to equal WORLD
    for static-CP. DiT collectives continue to use the model's NCCL lane group.
    A completed request is replaced immediately without opening an EPAC pulse.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        control_group: object | None,
        claim: Callable[[int], LaneWorkItem | None],
        execute: Callable[[LaneWorkItem, tuple[int, ...]], LaneWorkResult],
        complete: Callable[[LaneWorkResult], None],
        should_stop: Callable[[], bool],
        idle_sleep_s: float = 0.01,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid full-world lane worker rank/world_size")
        if idle_sleep_s <= 0:
            raise ValueError("idle_sleep_s must be positive")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.control_group = control_group
        self._claim = claim
        self._execute = execute
        self._complete = complete
        self._should_stop = should_stop
        self._idle_sleep_s = float(idle_sleep_s)
        self.ranks = tuple(range(self.world_size))

    def run(self) -> None:
        while True:
            command = [None]
            if self.rank == 0:
                if self._should_stop():
                    command[0] = {"action": "stop"}
                else:
                    item = self._claim(0)
                    command[0] = (
                        {"action": "idle"}
                        if item is None
                        else {"action": "run", "item": item}
                    )
            if self.world_size > 1:
                dist.broadcast_object_list(
                    command,
                    src=0,
                    group=self.control_group,
                    device=torch.device("cpu"),
                )
            payload = command[0] or {}
            action = payload.get("action")
            if action == "stop":
                return
            if action == "idle":
                time.sleep(self._idle_sleep_s)
                continue
            item = payload.get("item")
            if action != "run" or not isinstance(item, LaneWorkItem):
                raise RuntimeError("full-world lane received an invalid command")
            local_result = self._execute_safely(item)
            gathered = [None] * self.world_size if self.rank == 0 else None
            if self.world_size > 1:
                dist.gather_object(
                    local_result,
                    object_gather_list=gathered,
                    dst=0,
                    group=self.control_group,
                )
            else:
                gathered = [local_result]
            if self.rank == 0:
                assert gathered is not None
                self._complete(self._merge_results(item.request_id, gathered))

    def _execute_safely(self, item: LaneWorkItem) -> LaneWorkResult:
        try:
            return self._execute(item, self.ranks)
        except BaseException as exc:
            return LaneWorkResult(
                request_id=item.request_id,
                error=str(exc),
                metadata={"lane_rank": self.rank},
            )

    @staticmethod
    def _merge_results(
        request_id: str, results: list[LaneWorkResult | None]
    ) -> LaneWorkResult:
        valid = [result for result in results if result is not None]
        errors = [result.error for result in valid if result.error is not None]
        leader = valid[0] if valid else None
        if leader is None:
            return LaneWorkResult(
                request_id=request_id, error="lane returned no result"
            )
        if errors:
            return LaneWorkResult(
                request_id=request_id,
                error="; ".join(errors),
                metadata=dict(leader.metadata),
            )
        return leader


class RankExchange(Protocol):
    def __call__(self, message: Any) -> Any: ...


class AsyncResultChannel:
    """Publish large lane results without blocking the GPU worker loop."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        control_group: object | None,
        complete: Callable[[LaneWorkResult], None],
        on_transfer: Callable[[LaneWorkResult, int, int], None] | None = None,
        send: Callable[[Any, int], None] | None = None,
        recv: Callable[[int], Any] | None = None,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid async result channel rank/world_size")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.control_group = control_group
        self._complete = complete
        self._on_transfer = on_transfer
        self._send_override = send
        self._recv_override = recv
        self._pending: queue.Queue[LaneWorkResult | None] = queue.Queue()
        self._errors: queue.Queue[BaseException] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("async result channel has already started")
        self._started = True
        if self.rank == 0:
            self._threads = [
                threading.Thread(
                    target=self._serve_peer,
                    args=(peer,),
                    name=f"epac-result-recv-{peer}",
                    daemon=True,
                )
                for peer in range(1, self.world_size)
            ]
        else:
            self._threads = [
                threading.Thread(
                    target=self._publish_remote,
                    name="epac-result-publish",
                    daemon=True,
                )
            ]
        for thread in self._threads:
            thread.start()

    def submit(self, result: LaneWorkResult) -> None:
        if not isinstance(result, LaneWorkResult):
            raise TypeError("async result channel requires a LaneWorkResult")
        if not self._started or self._closed:
            raise RuntimeError("async result channel is not accepting results")
        self._raise_error()
        if self.rank == 0:
            started_ns = time.time_ns()
            self._complete(result)
            self._record_transfer(result, started_ns)
            return
        self._pending.put_nowait(result)

    def close(self) -> None:
        if not self._started or self._closed:
            return
        self._closed = True
        if self.rank != 0:
            self._pending.put(None)
        for thread in self._threads:
            thread.join()
        self._raise_error()

    def _publish_remote(self) -> None:
        try:
            while True:
                result = self._pending.get()
                if result is None:
                    self._send({"kind": "result_channel_close"}, 0)
                    response = self._recv(0)
                    if response.get("action") != "result_channel_close_ack":
                        raise RuntimeError(
                            "async result channel received an invalid close ack"
                        )
                    return
                started_ns = time.time_ns()
                self._send({"kind": "lane_result", "result": result}, 0)
                response = self._recv(0)
                if (
                    response.get("action") != "result_ack"
                    or response.get("request_id") != result.request_id
                ):
                    raise RuntimeError(
                        "async result channel received an invalid result ack"
                    )
                self._record_transfer(result, started_ns)
        except BaseException as exc:
            self._errors.put(exc)

    def _serve_peer(self, peer: int) -> None:
        try:
            while True:
                message = self._recv(peer)
                kind = message.get("kind")
                if kind == "result_channel_close":
                    self._send({"action": "result_channel_close_ack"}, peer)
                    return
                result = message.get("result")
                if kind != "lane_result" or not isinstance(result, LaneWorkResult):
                    raise RuntimeError(
                        "async result channel received an invalid message"
                    )
                self._complete(result)
                self._send(
                    {"action": "result_ack", "request_id": result.request_id},
                    peer,
                )
        except BaseException as exc:
            self._errors.put(exc)

    def _send(self, value: Any, destination: int) -> None:
        if self._send_override is not None:
            self._send_override(value, destination)
            return
        dist.send_object_list(
            [value],
            dst=destination,
            group=self.control_group,
            device=torch.device("cpu"),
        )

    def _recv(self, source: int) -> Any:
        if self._recv_override is not None:
            return self._recv_override(source)
        value = [None]
        dist.recv_object_list(
            value,
            src=source,
            group=self.control_group,
            device=torch.device("cpu"),
        )
        return value[0]

    def _record_transfer(self, result: LaneWorkResult, started_ns: int) -> None:
        if self._on_transfer is not None:
            self._on_transfer(result, started_ns, time.time_ns())

    def _raise_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("async result channel failed") from error


class DistributedRankExchange:
    """Point-to-point rank-0 control transport for dynamic lane workers.

    The rank-0 handler may block while it aggregates reports for one lane or
    pulse. Other lane handlers continue independently in their own threads.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        control_group: object | None,
        handler: Callable[[int, Any], Any] | None,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank exchange rank/world_size")
        if rank == 0 and handler is None:
            raise ValueError("rank 0 requires an exchange handler")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.control_group = control_group
        self._handler = handler
        self._errors: queue.Queue[BaseException] = queue.Queue()

    def run(self, worker: Callable[[RankExchange], None]) -> None:
        listeners = []
        if self.rank == 0:
            listeners = [
                threading.Thread(
                    target=self._serve_peer,
                    args=(peer,),
                    name=f"epac-rank-exchange-{peer}",
                    daemon=True,
                )
                for peer in range(1, self.world_size)
            ]
            for thread in listeners:
                thread.start()
        try:
            worker(self.exchange)
        finally:
            if self.rank != 0:
                self._send({"kind": "worker_exit"})
                self._recv()
            else:
                for thread in listeners:
                    thread.join()
                self._raise_peer_error()

    def exchange(self, message: Any) -> Any:
        if self.rank == 0:
            assert self._handler is not None
            return self._handler(0, message)
        self._send({"kind": "exchange", "payload": message})
        return self._recv()

    def _serve_peer(self, peer: int) -> None:
        try:
            while True:
                envelope = self._recv(src=peer)
                if envelope.get("kind") == "worker_exit":
                    self._send({"action": "exit_ack"}, dst=peer)
                    return
                if envelope.get("kind") != "exchange":
                    raise RuntimeError("rank exchange received an invalid envelope")
                assert self._handler is not None
                response = self._handler(peer, envelope.get("payload"))
                self._send(response, dst=peer)
        except BaseException as exc:
            self._errors.put(exc)

    def _send(self, value: Any, dst: int = 0) -> None:
        dist.send_object_list(
            [value],
            dst=dst,
            group=self.control_group,
            device=torch.device("cpu"),
        )

    def _recv(self, src: int = 0) -> Any:
        value = [None]
        dist.recv_object_list(
            value,
            src=src,
            group=self.control_group,
            device=torch.device("cpu"),
        )
        return value[0]

    def _raise_peer_error(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("rank exchange peer failed") from error

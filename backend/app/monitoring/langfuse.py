"""Langfuse 모니터링 통합.

Langfuse SDK를 래핑하여 트레이싱/스코어링 기능을 제공한다.
키 미설정 시 모든 호출이 no-op으로 동작한다.
"""
from __future__ import annotations

from typing import Any

from langfuse import Langfuse


class _NoOp:
    """어떤 메서드 호출이든 자기 자신을 반환하는 no-op 객체."""

    def __getattr__(self, name: str) -> Any:
        return self._noop

    def _noop(self, *args: Any, **kwargs: Any) -> "_NoOp":
        return self


_NOOP = _NoOp()


class LangfuseMonitor:
    """Langfuse SDK 래퍼.

    public_key/secret_key가 모두 설정되면 실제 Langfuse 인스턴스를 생성하고,
    하나라도 없으면 모든 호출이 no-op으로 동작한다.
    """

    def __init__(
        self,
        public_key: str | None,
        secret_key: str | None,
        host: str = "http://localhost:3100",
    ) -> None:
        self.enabled = bool(public_key and secret_key)
        if self.enabled:
            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
        else:
            self._langfuse = None

    def create_trace(self, name: str, input: str) -> Any:
        if not self.enabled:
            return _NOOP
        try:
            if hasattr(self._langfuse, "trace"):
                return self._langfuse.trace(name=name, input=input)
            if hasattr(self._langfuse, "start_span"):
                span = self._langfuse.start_span(name=name, input={"query": input})
                # v3 SDK는 trace를 span으로 시작하므로 trace 메타를 함께 업데이트한다.
                if hasattr(span, "update_trace"):
                    try:
                        span.update_trace(name=name, input={"query": input})
                    except Exception:
                        pass
                return span
        except Exception:
            return _NOOP
        return _NOOP

    def end_observation(self, observation: Any, output: Any | None = None) -> None:
        if not self.enabled or observation is None:
            return
        try:
            if output is not None and hasattr(observation, "update"):
                observation.update(output=output)
        except Exception:
            pass
        try:
            if hasattr(observation, "end"):
                observation.end()
        except Exception:
            pass

    def end_trace(
        self,
        trace: Any,
        *,
        name: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
    ) -> None:
        if not self.enabled or trace is None:
            return

        updates: dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if input is not None:
            updates["input"] = input
        if output is not None:
            updates["output"] = output

        if updates and hasattr(trace, "update_trace"):
            try:
                trace.update_trace(**updates)
            except Exception:
                pass

        # trace 객체인 경우 update()로도 루트 필드를 업데이트할 수 있다.
        if updates and hasattr(trace, "update"):
            try:
                trace.update(**updates)
            except Exception:
                pass

        try:
            if hasattr(trace, "end"):
                trace.end()
        except Exception:
            pass

    def create_span(self, trace: Any, name: str) -> Any:
        if not self.enabled:
            return _NOOP
        try:
            if hasattr(trace, "span"):
                return trace.span(name=name)
            if hasattr(trace, "start_span"):
                return trace.start_span(name=name)
            if hasattr(trace, "start_observation"):
                return trace.start_observation(name=name, as_type="span")
        except Exception:
            return _NOOP
        return _NOOP

    def create_generation(
        self, trace: Any, name: str, model: str, input: dict,
    ) -> Any:
        if not self.enabled:
            return _NOOP
        try:
            if hasattr(trace, "generation"):
                return trace.generation(name=name, model=model, input=input)
            if hasattr(trace, "start_generation"):
                return trace.start_generation(name=name, model=model, input=input)
            if hasattr(trace, "start_observation"):
                return trace.start_observation(
                    name=name, as_type="generation", model=model, input=input,
                )
        except Exception:
            return _NOOP
        return _NOOP

    def score(self, trace_id: str, name: str, value: float) -> None:
        if not self.enabled:
            return
        if not trace_id:
            return
        try:
            if hasattr(self._langfuse, "score"):
                self._langfuse.score(trace_id=trace_id, name=name, value=value)
                return
            if hasattr(self._langfuse, "create_score"):
                self._langfuse.create_score(trace_id=trace_id, name=name, value=value)
                return
        except Exception:
            return

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self._langfuse.flush()
        except Exception:
            return

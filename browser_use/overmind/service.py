"""Overmind trace export for browser-use.

Initialises the Overmind SDK from PyPI when ``OVERMIND_API_KEY`` is set, fans out
onto an existing OpenTelemetry ``TracerProvider`` when one is already installed
(e.g. by Laminar / Traceloop), and exposes helpers for the agent lifecycle
(conversation id, final result, flush).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Literal

logger = logging.getLogger(__name__)

_overmind_module: Any | None = None
_initialised = False


def _load_overmind() -> Any | None:
	global _overmind_module
	if _overmind_module is not None:
		return _overmind_module
	if importlib.util.find_spec('overmind') is None:
		return None
	import overmind as om

	_overmind_module = om
	return om


def is_enabled() -> bool:
	"""True when Overmind should export traces for this process."""
	if os.environ.get('OVERMIND_DISABLED', '').lower() in {'1', 'true', 'yes', 'on'}:
		return False
	return bool(os.environ.get('OVERMIND_API_KEY', '').strip()) and _load_overmind() is not None


def _detect_llm_providers() -> list[str]:
	"""Return installed LLM SDKs that Overmind can auto-instrument."""
	providers: list[str] = []
	for name, module in (
		('openai', 'openai'),
		('anthropic', 'anthropic'),
		('google', 'google.genai'),
	):
		if importlib.util.find_spec(module) is not None:
			providers.append(name)
	return providers


def _fan_out_or_init(overmind: Any, *, providers: list[str]) -> None:
	"""Greenfield init, or attach Overmind exporter to an existing provider."""
	from opentelemetry import trace
	from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
	from opentelemetry.sdk.trace import TracerProvider
	from opentelemetry.sdk.trace.export import BatchSpanProcessor

	from overmind.tracing import enable_tracing, get_api_settings

	service_name = os.environ.get('OVERMIND_SERVICE_NAME', 'browser-use')
	agent_name = os.environ.get('OVERMIND_AGENT_NAME', 'Browser Use Agent')
	agent_id = os.environ.get('OVERMIND_AGENT_ID')
	project_id = os.environ.get('OVERMIND_PROJECT_ID')
	environment = os.environ.get('OVERMIND_ENVIRONMENT') or os.environ.get('ENVIRONMENT')

	provider = trace.get_tracer_provider()
	if isinstance(provider, TracerProvider):
		api_key, base_url = get_api_settings()
		provider.add_span_processor(
			BatchSpanProcessor(
				OTLPSpanExporter(
					endpoint=f'{base_url}/api/v1/traces',
					headers={'X-Api-Key': api_key},
				)
			)
		)
		enable_tracing(providers)
		if agent_id:
			overmind.set_agent_id(agent_id)
		if agent_name:
			overmind.set_agent_name(agent_name)
		if project_id:
			overmind.set_project_id(project_id)
		logger.info('Overmind telemetry attached to existing TracerProvider (fan-out mode)')
		return

	overmind.init(
		service_name=service_name,
		agent_name=agent_name,
		agent_id=agent_id,
		project_id=project_id,
		environment=environment,
		providers=providers,
	)
	logger.info('Overmind telemetry initialised (service=%s, agent=%s)', service_name, agent_name)


def init_if_configured(*, conversation_id: str | None = None, task: str | None = None) -> bool:
	"""Initialise Overmind once when configured. Returns True if active."""
	global _initialised
	if _initialised or not is_enabled():
		return _initialised

	overmind = _load_overmind()
	assert overmind is not None

	providers = _detect_llm_providers()
	_fan_out_or_init(overmind, providers=providers)

	if conversation_id:
		overmind.set_conversation_id(conversation_id)
	if task:
		overmind.set_workflow_name(task[:256])

	_initialised = True
	return True


def flush_traces(timeout_millis: int = 5000) -> None:
	if not _initialised:
		return
	overmind = _load_overmind()
	if overmind is None:
		return
	try:
		overmind.force_flush_traces(timeout_millis=timeout_millis)
	except Exception:
		logger.debug('Overmind trace flush failed', exc_info=True)


def deliver_agent_result(result: Any) -> None:
	"""Record the terminal agent result on a child span (PyPI tracing model)."""
	if not _initialised:
		return
	overmind = _load_overmind()
	if overmind is None:
		return
	try:
		with overmind.start_span('agent.final_result', span_type=overmind.SpanType.FUNCTION):
			overmind.set_tag('outputs', result)
	except Exception:
		logger.debug('Overmind final result capture failed', exc_info=True)


OvermindSpanKind = Literal['entry_point', 'workflow', 'tool', 'function', 'retrieval']


class OvermindTelemetry:
	"""Thin facade used by :mod:`browser_use.observability`."""

	@staticmethod
	def enabled() -> bool:
		return is_enabled()

	@staticmethod
	def span_kind_for_name(name: str | None) -> OvermindSpanKind:
		span_name = name or ''
		if span_name in {'agent.run'}:
			return 'entry_point'
		if span_name in {'agent.step'}:
			return 'workflow'
		if span_name.startswith('chat_') or span_name.endswith('_ainvoke'):
			return 'function'
		return 'function'

	@staticmethod
	def should_trace_debug_span(name: str | None) -> bool:
		"""Interior spans that should always export when Overmind is on."""
		return (name or '') in {
			'get_next_action',
			'get_model_output',
			'act',
			'multi_act',
			'_get_next_action',
			'_execute_actions',
		}

	@staticmethod
	def decorator(kind: OvermindSpanKind, name: str | None):
		overmind = _load_overmind()
		if overmind is None:
			raise RuntimeError('overmind is not installed')

		span_name = name
		if kind == 'entry_point':
			return overmind.entry_point(name=span_name)
		if kind == 'workflow':
			return overmind.workflow(name=span_name)
		if kind == 'tool':
			return overmind.tool(name=span_name)
		if kind == 'retrieval':
			return overmind.retrieval(name=span_name)
		return overmind.function(name=span_name)

	@staticmethod
	def tool_span(name: str):
		"""Context manager for per-action tool spans inside ``Tools.act``."""
		overmind = _load_overmind()
		if overmind is None or not _initialised:
			from contextlib import nullcontext

			return nullcontext()

		return overmind.start_span(name, span_type=overmind.SpanType.TOOL)

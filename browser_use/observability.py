# @file purpose: Observability module for browser-use that handles optional lmnr integration with debug mode support
"""
Observability module for browser-use

This module provides observability decorators that optionally integrate with lmnr (Laminar) for tracing.
If lmnr is not installed, it provides no-op wrappers that accept the same parameters.

When ``OVERMIND_API_KEY`` is set, the same decorators also emit spans to Overmind via the local
Overmind SDK (trajectory branch), including ``code.namespace`` / ``code.function.name`` anchors for
trace binding and live scoring.

Features:
- Optional lmnr integration - works with or without lmnr installed
- Optional Overmind integration - works when OVERMIND_API_KEY is set
- Debug mode support - observe_debug only traces when in debug mode (lmnr); Overmind still traces
  named interior spans (get_model_output, act, multi_act, …) regardless of debug mode
- Full parameter compatibility with lmnr observe decorator
- No-op fallbacks when lmnr is unavailable
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, TypeVar, cast

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

# Type definitions
F = TypeVar('F', bound=Callable[..., Any])


# Check if we're in debug mode
def _is_debug_mode() -> bool:
	"""Check if we're in debug mode based on environment variables or logging level."""

	lmnr_debug_mode = os.getenv('LMNR_LOGGING_LEVEL', '').lower()
	if lmnr_debug_mode == 'debug':
		# logger.info('Debug mode is enabled for observability')
		return True
	# logger.info('Debug mode is disabled for observability')
	return False


# Try to import lmnr observe
_LMNR_AVAILABLE = False
_lmnr_observe = None

try:
	from lmnr import observe as _lmnr_observe  # type: ignore

	if os.environ.get('BROWSER_USE_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
		logger.debug('Lmnr is available for observability')
	_LMNR_AVAILABLE = True
except (ImportError, TypeError):
	if os.environ.get('BROWSER_USE_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
		logger.debug('Lmnr is not available for observability')
	_LMNR_AVAILABLE = False


def _overmind_enabled() -> bool:
	try:
		from browser_use.overmind.service import is_enabled

		return is_enabled()
	except Exception:
		return False


def _ensure_overmind_initialized() -> bool:
	"""Lazy-init Overmind on first traced call (decorators run before agent.run body)."""
	if not _overmind_enabled():
		return False
	try:
		from browser_use.overmind.service import init_if_configured

		return init_if_configured()
	except Exception:
		logger.debug('Overmind lazy init failed', exc_info=True)
		return False


def _stack_overmind_decorator(
	func: F,
	*,
	name: str | None,
	kind: Literal['entry_point', 'workflow', 'tool', 'function', 'retrieval'] | None = None,
) -> F:
	if not _overmind_enabled():
		return func

	try:
		if not _ensure_overmind_initialized():
			return func

		from browser_use.overmind.service import OvermindTelemetry

		span_kind = kind or OvermindTelemetry.span_kind_for_name(name)
		decorator = OvermindTelemetry.decorator(span_kind, name)
		return cast(F, decorator(func))
	except Exception:
		logger.debug('Overmind decorator could not be applied to %s', name, exc_info=True)
		return func


def _create_no_op_decorator(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	**kwargs: Any,
) -> Callable[[F], F]:
	"""Create a no-op decorator that accepts all lmnr observe parameters but does nothing."""
	import asyncio

	def decorator(func: F) -> F:
		if asyncio.iscoroutinefunction(func):

			@wraps(func)
			async def async_wrapper(*args, **kwargs):
				return await func(*args, **kwargs)

			return cast(F, async_wrapper)
		else:

			@wraps(func)
			def sync_wrapper(*args, **kwargs):
				return func(*args, **kwargs)

			return cast(F, sync_wrapper)

	return decorator


def observe(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT',
	overmind_kind: Literal['entry_point', 'workflow', 'tool', 'function', 'retrieval'] | None = None,
	**kwargs: Any,
) -> Callable[[F], F]:
	"""
	Observability decorator that traces function execution when lmnr is available,
	and/or when Overmind is configured via ``OVERMIND_API_KEY``.

	This decorator will use lmnr's observe decorator if lmnr is installed,
	otherwise it will be a no-op that accepts the same parameters.

	Args:
	    name: Name of the span/trace
	    ignore_input: Whether to ignore function input parameters in tracing
	    ignore_output: Whether to ignore function output in tracing
	    metadata: Additional metadata to attach to the span
	    overmind_kind: Explicit Overmind span type (entry_point, workflow, tool, function, retrieval)
	    **kwargs: Additional parameters passed to lmnr observe

	Returns:
	    Decorated function that may be traced depending on lmnr availability

	Example:
	    @observe(name="my_function", metadata={"version": "1.0"})
	    def my_function(param1, param2):
	        return param1 + param2
	"""
	lmnr_kwargs = {
		'name': name,
		'ignore_input': ignore_input,
		'ignore_output': ignore_output,
		'metadata': metadata,
		'span_type': span_type,
		'tags': ['observe', 'observe_debug'],  # important: tags need to be created on laminar first
		**kwargs,
	}

	def decorator(func: F) -> F:
		wrapped = func
		wrapped = _stack_overmind_decorator(wrapped, name=name, kind=overmind_kind)

		if _LMNR_AVAILABLE and _lmnr_observe:
			return cast(F, _lmnr_observe(**lmnr_kwargs)(wrapped))
		return _create_no_op_decorator(**lmnr_kwargs)(wrapped)

	return decorator


def observe_debug(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT',
	overmind_kind: Literal['entry_point', 'workflow', 'tool', 'function', 'retrieval'] | None = None,
	**kwargs: Any,
) -> Callable[[F], F]:
	"""
	Debug-only observability decorator that only traces when in debug mode.

	This decorator will use lmnr's observe decorator if both lmnr is installed
	AND we're in debug mode, otherwise it will be a no-op.

	When Overmind is configured, interior spans listed in
	:meth:`OvermindTelemetry.should_trace_debug_span` are traced regardless of
	debug mode so the call graph carries ``code.namespace`` anchors.

	Debug mode is determined by:
	- DEBUG environment variable set to 1/true/yes/on
	- BROWSER_USE_DEBUG environment variable set to 1/true/yes/on
	- Root logging level set to DEBUG or lower

	Args:
	    name: Name of the span/trace
	    ignore_input: Whether to ignore function input parameters in tracing
	    ignore_output: Whether to ignore function output in tracing
	    metadata: Additional metadata to attach to the span
	    overmind_kind: Explicit Overmind span type
	    **kwargs: Additional parameters passed to lmnr observe

	Returns:
	    Decorated function that may be traced only in debug mode

	Example:
	    @observe_debug(ignore_input=True, ignore_output=True,name="debug_function", metadata={"debug": True})
	    def debug_function(param1, param2):
	        return param1 + param2
	"""
	lmnr_kwargs = {
		'name': name,
		'ignore_input': ignore_input,
		'ignore_output': ignore_output,
		'metadata': metadata,
		'span_type': span_type,
		'tags': ['observe_debug'],  # important: tags need to be created on laminar first
		**kwargs,
	}

	def decorator(func: F) -> F:
		wrapped = func
		if _overmind_enabled():
			try:
				if _ensure_overmind_initialized():
					from browser_use.overmind.service import OvermindTelemetry

					if OvermindTelemetry.should_trace_debug_span(name) or _is_debug_mode():
						wrapped = _stack_overmind_decorator(wrapped, name=name, kind=overmind_kind)
			except Exception:
				logger.debug('Overmind debug decorator could not be applied to %s', name, exc_info=True)

		if _LMNR_AVAILABLE and _lmnr_observe and _is_debug_mode():
			return cast(F, _lmnr_observe(**lmnr_kwargs)(wrapped))
		return _create_no_op_decorator(**lmnr_kwargs)(wrapped)

	return decorator


# Convenience functions for checking availability and debug status
def is_lmnr_available() -> bool:
	"""Check if lmnr is available for tracing."""
	return _LMNR_AVAILABLE


def is_overmind_available() -> bool:
	"""Check if Overmind tracing is configured for this process."""
	return _overmind_enabled()


def is_debug_mode() -> bool:
	"""Check if we're currently in debug mode."""
	return _is_debug_mode()


def get_observability_status() -> dict[str, bool]:
	"""Get the current status of observability features."""
	return {
		'lmnr_available': _LMNR_AVAILABLE,
		'overmind_available': _overmind_enabled(),
		'debug_mode': _is_debug_mode(),
		'observe_active': _LMNR_AVAILABLE or _overmind_enabled(),
		'observe_debug_active': (_LMNR_AVAILABLE and _is_debug_mode()) or _overmind_enabled(),
	}

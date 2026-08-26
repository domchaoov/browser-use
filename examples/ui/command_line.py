"""
To Use It:

Example 1: Auto-detect LLM from .env (prefers OpenRouter if OPENROUTER_API_KEY is set)
python command_line.py

Example 2: Custom query
python command_line.py --query "go to google and search for browser-use"

Example 3: Pick a provider explicitly
python command_line.py --query "find latest Python tutorials" --provider openrouter
python command_line.py --query "find latest Python tutorials" --provider anthropic

Run from anywhere in the repo — .env is loaded from the repository root.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure local repository (browser_use) is accessible
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / '.env')

from browser_use import Agent
from browser_use.browser import BrowserSession
from browser_use.tools.service import Tools


def _require_env(name: str) -> str:
	value = os.getenv(name, '').strip()
	if not value:
		raise ValueError(f'Error: {name} is not set. Add it to {REPO_ROOT / ".env"}')
	return value


def detect_provider() -> str:
	"""Pick the first configured provider from the environment."""
	for provider, env_var in (
		('openrouter', 'OPENROUTER_API_KEY'),
		('openai', 'OPENAI_API_KEY'),
		('anthropic', 'ANTHROPIC_API_KEY'),
		('google', 'GOOGLE_API_KEY'),
		('browser-use', 'BROWSER_USE_API_KEY'),
	):
		if os.getenv(env_var, '').strip():
			return provider
	raise ValueError(
		f'No LLM API key found in {REPO_ROOT / ".env"}. '
		'Set one of: OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, BROWSER_USE_API_KEY'
	)


def get_llm(provider: str):
	if provider == 'openrouter':
		from browser_use import ChatOpenRouter

		return ChatOpenRouter(
			# model=os.getenv('OPENROUTER_MODEL', 'openai/gpt-4.1-mini'),
			model=os.getenv('OPENROUTER_MODEL', 'moonshotai/kimi-k3'),
			api_key=_require_env('OPENROUTER_API_KEY'),
			temperature=0.0,
		)
	if provider == 'anthropic':
		from browser_use.llm import ChatAnthropic

		return ChatAnthropic(
			model=os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-0'),
			api_key=_require_env('ANTHROPIC_API_KEY'),
			temperature=0.0,
		)
	if provider == 'openai':
		from browser_use import ChatOpenAI

		return ChatOpenAI(
			model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
			api_key=_require_env('OPENAI_API_KEY'),
			temperature=0.0,
		)
	if provider == 'google':
		from browser_use import ChatGoogle

		return ChatGoogle(model=os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash'))
	if provider == 'browser-use':
		from browser_use import ChatBrowserUse

		return ChatBrowserUse(model=os.getenv('BROWSER_USE_MODEL', 'bu-2-0-mini-preview'))

	raise ValueError(f'Unsupported provider: {provider}')


def parse_arguments():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
	parser.add_argument(
		'--query', type=str, help='The query to process', default='go to reddit and search for posts about browser-use'
	)
	parser.add_argument(
		'--provider',
		type=str,
		choices=['auto', 'openrouter', 'openai', 'anthropic', 'google', 'browser-use'],
		default='auto',
		help='LLM provider (default: auto — first key found in .env)',
	)
	return parser.parse_args()


def initialize_agent(query: str, provider: str):
	"""Initialize the browser agent with the given query and provider."""
	resolved_provider = detect_provider() if provider == 'auto' else provider
	print(f'Using provider: {resolved_provider}')
	llm = get_llm(resolved_provider)
	tools = Tools()
	browser_session = BrowserSession()

	return Agent(
		task=query,
		llm=llm,
		tools=tools,
		browser_session=browser_session,
		use_vision=True,
		max_actions_per_step=1,
	), browser_session


async def main():
	"""Main async function to run the agent."""
	args = parse_arguments()
	agent, browser_session = initialize_agent(args.query, args.provider)

	await agent.run(max_steps=25)

	input('Press Enter to close the browser...')
	await browser_session.kill()


if __name__ == '__main__':
	asyncio.run(main())

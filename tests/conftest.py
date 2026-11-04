"""Test setup: never let a test pick up real API keys from .env or call a real LLM."""
import os

os.environ["LLM_PROVIDER"] = "mock"  # set before any app starts; config.load_env() never overrides real variables

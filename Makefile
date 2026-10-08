clean:
	find . \( -path './.git' -o -path './SpikeLoRA' -o -path './peft' -o -path './results' -o -path './logs' \) -prune -o -type d -name '__pycache__' -print0 | xargs -0 rm -rf
	find . \( -path './.git' -o -path './SpikeLoRA' -o -path './peft' -o -path './results' -o -path './logs' \) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -print0 | xargs -0 rm -f
	find . \( -path './.git' -o -path './SpikeLoRA' -o -path './peft' -o -path './results' -o -path './logs' \) -prune -o -type d -name '*.egg-info' -print0 | xargs -0 rm -rf
	rm -rf .pytest_cache .mypy_cache build dist

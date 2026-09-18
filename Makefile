.PHONY: test figures slides evidence

test:
	python -m pytest -q -s

figures:
	python -m fusion.run --phase 1
	python -m fusion.run --phase 4

slides:
	python -m fusion.run --phase 5

evidence:
	python tools/check_weight_tails.py
	python tools/twin_default_run.py

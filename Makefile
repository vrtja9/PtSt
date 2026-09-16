.PHONY: test figures slides

test:
	pytest -q

figures:
	python -m fusion.run --phase 1
	python -m fusion.run --phase 4

slides:
	python -m fusion.run --phase 5

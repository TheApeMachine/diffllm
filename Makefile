.PHONY: generate

generate:
	python3 generate.py \
		--checkpoint checkpoints/model_epoch001.pt \
		--prompt "def fibonacci(n):" \
		--num-samples 5 \
		--sampler ddim \
		--ddim-steps 50 \
		--guidance-scale 7.5
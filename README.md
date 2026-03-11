A simple JAX reproduction of the [Tiny Recursion Model](https://arxiv.org/abs/2510.04871) (TRM) trained on the sudoku-extreme task. Attempting to match the paper's results and then perform a few experiments. See the official PyTorch implementation [here](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

Thanks to the [TPU Research Cloud](https://sites.research.google/trc/about/) program for the compute!

Running with 

```bash
uv run --with jax\[tpu\] main.py --workdir "logs/vanilla" --seed 0 \
	--max_checkpoints 1 --N_sup_eval_mult 2.0

# loads best checkpoint and evals up to 16*32 = 512
uv run --with jax\[tpu\] main.py --workdir "logs/vanilla" --eval_only --N_sup_eval_mult 32.0
```

currently yields this run ([checkpoint](https://huggingface.co/emiliocantuc/trm-vanilla-2)), which displays test-time scaling. I.e. after training we measure % of sudoku puzzles solved (right plot below) as we increase supervision steps $N_\text{sup}$:

<img width="815" height="432" alt="Screenshot 2026-02-07 at 6 43 14 PM" src="https://github.com/user-attachments/assets/a0745502-e905-4c18-b51f-8543dd6e23c5" />


### Todo
- [ ] make val set more representative of test
- [ ] update new checkpoints and results w/test-time scaling curves
- [ ] small runs with test-time aug (rand init state + bigger N_sups)

### Citations

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

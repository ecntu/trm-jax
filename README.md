A simple JAX reproduction of the [Tiny Recursion Model](https://arxiv.org/abs/2510.04871) (TRM) trained on the sudoku-extreme task. Attempting to match the paper's results and then perform a few experiments. See the official PyTorch implementation [here](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

Thanks to the [TPU Research Cloud](https://sites.research.google/trc/about/) program for the compute!

Run with: 

```bash
uv run --with jax\[tpu\] main.py --workdir logs/run_name
```
### Todo
- [ ] Match paper performance (>85% solve rate)
- [ ] Fix random seed determinism

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
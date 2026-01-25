A simple JAX reproduction of the [Tiny Recursion Model](https://arxiv.org/abs/2510.04871) (TRM) trained on the sudoku-extreme task. Attempting to match the paper's results and then perform a few experiments. See the official PyTorch implementation [here](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

Thanks to the [TPU Research Cloud](https://sites.research.google/trc/about/) program for the compute!

Running with 

```bash
uv run --with jax\[tpu\] main.py --workdir logs/run_name
```

currently yields this run and [checkpoint](https://huggingface.co/emiliocantuc/trm_test):

<img width="1256" height="428" alt="Screenshot 2026-01-20 at 9 19 44 AM" src="https://github.com/user-attachments/assets/52d83aaf-4794-4879-a6b2-85a15d0f97e9" />


### Todo
- [ ] Match paper performance (>85% solve rate)

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

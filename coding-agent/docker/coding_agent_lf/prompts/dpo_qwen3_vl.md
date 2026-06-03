Your task is to iteratively produce an `instruction.md` that contains only the information necessary for a testee coding agent to convert a given PyTorch repository into JAX and pass the evaluator.

## Paths

- Evaluator path: `eval`
- Source PyTorch repository: `dpo/qwen3_vl/torch_orig`
- Prepared `python` environment for you and the testee: the active `lf` environment (`python` after `conda activate lf`)
- Attempt workspace: `.attempt/attempt_{idx}/` (e.g., attempt_1, attempt_2, etc)

## Goal

Create an `instruction.md` such that a testee model, using only the contents of that `instruction.md` as its task guidance, can convert the source PyTorch repository into a JAX repository that passes all of the evaluator test.

## Required loop

You must work in an iterative loop:

1. **COMPLETELY investigate** the evaluator path and the source repository until you understand any details required for passing the test.
2. Distill the goal, investigation result, and may discovery from previous error message (if applicable) into `instruction.md`.
3. Directly input that `instruction.md` to a testee coding agent (in your case, the call client_subagent), specify the `workspace` as the attempt path, and enable the `restricted mode`.
4. Copy the files from torch repo into the attempt workspace for testee and let the testee attempt the PyTorch-to-JAX conversion using only `instruction.md`.
5. After the testee return to you, run the evaluator on the produced repository.
6. Analysis the error from evaluator. (1) If the error belongs to the evaluator's implementation, fix the evaluator. (2) Otherwise, fix the testee's implementation. Repeat the two sub-steps until all tests pass in this attempt.
7. Distill your fixing experience into `instruction.md` and start a new attempt.
8. Repeat until the testee can pass all evaluator tests using only `instruction.md`.

- When you update the instruction, let the testee work in a new attempt folder.
- Put the instruction of every attempt into the corresponding attempt folder.

## Hard constraints

- `instruction.md` must describe only the necessary acceptance conditions and implementation-critical requirements needed for passing the evaluation.
- Do not turn `instruction.md` into a step-by-step tutorial for the testee.
- Do not include evaluator paths, evaluator commands, config filenames, or how the evaluator is invoked.
- Do not include any hidden side-channel knowledge that the testee would not have from `instruction.md`.
- The pass condition is: the testee succeeds because `instruction.md` is sufficient, not because the evaluator was leaked or because extra manual intervention happened outside that file.

## Deliverable

Your final deliverable is the best `instruction.md` you can iteratively refine under this loop, together with a concise report of:
- whether the final testee attempt passed,
- the main evaluator-driven refinements you made,
- and any remaining blocker if full success was not reached.

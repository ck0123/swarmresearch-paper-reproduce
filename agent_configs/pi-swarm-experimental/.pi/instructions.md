# Instructions

You are a researcher working on an open research problem as part of a larger community. They are continually experimenting and updating the shared, git-based experiment history alongside you.

## Executing an experiment
Execute exactly one experiment. Running the evaluator and adding a commit message marks the experiment's end.

### STEP 1: Study the git history to stay up-to-date with new findings 
Use `git log --graph --all --format='%h %d %s [%an]'`

### STEP 2: Ideate your next experiment and setup + checkout its branch
- After deciding on an idea, decide on your experiment’s parent experiments and choose the correct branch structure:
  - Select the parent experiment whose commit your experiment directly builds on.
    - *(Definition)* An experiment *A* builds on *B* if *A* directly reuses or modifies concrete elements from *B*, such as its code, methods, intermediate artifacts, or findings/analysis.
    - Broad inspiration or similarity (e.g., adopting a general idea without using any of *B*’s concrete elements) does not count as building on *B*.
  - If your experiment is a direct refinement of the current tip of an existing branch, check that branch out and commit the new experiment to that same branch.
  - If your experiment builds on an earlier commit in a branch and needs to diverge from that line, create a new branch from that commit.
  - If your experiment builds on multiple experiments, represent this with a merge commit. For example, if experiment *C* combines elements of *A* and *B*:
    ```
    git checkout -b C A
    git merge --no-commit B
    ```
    - Use `--no-commit` so the history contains only commits corresponding to experiments, not intermediate merge-only commits.
    - The single commit you make afterward is the experiment commit.
    - Merges are strictly used to track that an experiment builds on multiple experiments. When resolving conflicts, you may freely modify content.
  - If your experiment uses a significantly new approach that diverges from all existing experiments, use `main/master` as the parent.
  - Make sure the correct branch is checked out before making your experiment commit.

Checkout the experiment branch before moving on. Do NOT do any implementation or testing until you checkout the branch.

### STEP 3: Implement the experiment
- Edit `initial_program.py` for your solution
- One idea per experiment. Implement the idea efficiently. 
- If you can't make it work after a few attempts, then give up immediately. Don't branch into broad exploratory investigations. Don't iterate forever.
- Don't do evals on your own. Use the standard evaluator

### STEP 4: Run the standard evaluator once
- You are only allowed to run the evaluator once
- Run exactly one final command:
  ```
  ./task-eval --commit-message "concise experiment description"
  ```
  - Do NOT run the evaluation with a timeout
  - Do NOT include "| score =" in your commit message. Just a concise description of the experiment.
- Do not run `git commit` manually. The evaluator will commit your workspace with the evaluated score.
- After the evaluator prints its output, do not continue, do not run any commands, and do not write any further messages. This marks the end of your experiment.

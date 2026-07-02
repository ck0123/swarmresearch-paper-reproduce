---
name: shepherd
description: How to organize a population of analysis agents
---

You organize a population of analysis agents towards performing analysis from diverse angles. The point is we want to study the problem from as many as different lenses as possible. Your goal is to manage the population to follow these goals:
- Initialize a diverse population of analysis 
- Maintain a diverse population of analysis and avoid analysis collapse throughout the discovery process
- Prioritize effort on promising analysis
- Make consistent progress and break out of plateaus of not gaining new information

You accomplish these goals by spawning analysis agents, selecting their parents, and adding minimal context as needed. *The most important rule*: analysis agents independently determine what they do; you are not allowed to instruct them to work on specific ideas. 

Keep going until you've exhausted your agent budget. The user assigns a budget in their initial prompt. If they don't, assume it's unlimited (never stop, go forever, don't ask for permission from the user). Keep count of how many you've spawned. Don’t give up, keep searching for new lenses or launching serial agents.

### Adapting analysis agent patterns
You should constantly think and reflect on current progress to select the right analysis agent pattern. Adapt the type of analysis agent, size of fan-outs, amount of serialization, etc. to current progress. Some examples of patterns:
- *Analysis Fan-out*: Multiple analysers from one parent. Use on plateaus to explore multiple promising parents. More concurrent analysers increases diversity. However, emphasize serializing analyser agents which strategically build on top of discoveries while concurrent analysers are unaware of each other. 
- *Serial analysers*: Select promising analyser parents from prior wave. Launch more analysers using them as parents. Actuate the number of analysers in each fan-out. 1 can be enough. You can add more after they're complete if not enough.
- *Multi-parent*: Merge multiple parent branches so the analysis agent can combine or take inspiration from multiple anlayses
- *Combinations*: Interleave serial analysers, fan-outs.

**Be creative. These are examples of ways to use the primitives available to you to make progress.**

### Searching for better families of approaches from initialization and on plateaus
**A substantial portion of your analyser budget should go towards building a large, diverse initial population** of ideas from master. Then, continue to spawn serial analysers ontop of promising diverse parents. Keep going until exploration plateaus.
- Launch explore agents without any added context in small batches since they might start repeating ideas. 
- If explore agents collapse onto similar ideas, nudge exploration by providing a summary of already explored angles and prompt to explore new angles as context. NEVER tell them which ideas to explore, just what not to explore again.

### Add minimal context as needed
analysis agents only have knowledge of their branch. They do NOT share your context and do NOT have context on other analysis agents. Only add context if necessary and keep it minimal and non-prescriptive. You might:
  - *Increase novelty*: Provide a summary of already explored ideas and strictly prompt them to explore new, more creative lenses.
  - *Improve strategy*: On plateaus, provide a 1) briefing of discoveries and 2) a suggestion to identify a novel analysis that fills in current blindspots, gaps, or pursues a more horizontal direction
  - *Share context*: Add context of relevant findings from other analysis agents that are helpful to refine their solution. Only add complementary findings. 
  - *Control the level of explorativenss*. If the agent should investigate from a radical lense, then say so. If it should explore smaller expansions, then say so. 

IMPORTANT GUARDRAIL: Do not directly suggest or instruct what idea to pursue. Just give them context that is useful for them to determine what's best on their own.

### Selecting parents by creating new branches
The neighborhood explored by analysis agents depends on their parent solution which serves as their solution's starting point. You define parent solutions by setting up the git branch they work inside.
- Spawn analysis agents on diverse parents. You don't know squat, so never collapse onto a single idea.
- Don't end ideas prematurely. If a analysis agent encounters a promusing idea that doesn't beat an idea you've already spent resources developing, consider refining and exploring it further. It might be a new neighborhood containing an improved solution.

Every agent must receive a branch that already exists. Always create a new non-descriptive branch name for each new agent like `researcher-17`. You control agents by selecting their parents, not by telling them what to do. 
- Single-parent experiment:
  ```
  git branch researcher-17 parent-branch-or-commit
  ```
  Use if an agent should build off a single solution.
- New approach from baseline:
  ```
  git branch researcher-17 master
  ```
  Use if an agent should explore new, from-scratch approaches.
  Use `main` instead of `master` if this repository uses `main`.
- Multi-parent experiment:
  ```
  mkdir -p .worktrees
  git worktree add -b researcher-17 .worktrees/researcher-17 first-parent
  cd .worktrees/researcher-17
  git merge --no-commit second-parent
  ```
  Use if an agent should combine multiple solutions.
  Add more parents with additional `git merge --no-commit` commands if needed.
  Merge conflicts are expected. A multi-parent branch is only valid if the worktree has no unresolved conflicts before launch. If there are conflicts, resolve them minimally:
  - keep one valid `initial_program.py` as the active evaluated program
  - save other parent implementations as reference files with unique names, such as `parent_researcher_8_initial_program.py`
  - combine or keep `findings.md`
  - commit the prepared setup:
    ```
    git add -A
    git commit -m "Prepare multi-parent researcher-17"
    ```

Do not implement, test, or edit experiment code while setting up branches. Branch setup only.

### Spawning agents
Spawn as many analysis agents as needed using both Claude and Codex non-interactive sessions. Follow their format exactly with every flag included. Include context in session prompts as needed. Each agent will return its session id along with a summary of its results. Launch an equal number of Claude and Codex agents. The overall priority is to select Claude or Codex based on which will increase idea diversity and have a higher chance of making a breakthrough.
- To spawn an `analyser`, start a fresh agent session:
  - Codex:
  ```
  tmp=$(mktemp); codex exec --dangerously-bypass-approvals-and-sandbox --json --output-last-message "$tmp" "Follow the shepherd analyst skill. Your branch name is branch_name. <optional: context>" | jq -r 'select(.type=="thread.started") | "session_id=\(.thread_id)"'; printf 'message=%s\n' "$(cat "$tmp")"; rm -f "$tmp"
  ```
  - Claude:
  ```
  BASH_DEFAULT_TIMEOUT_MS=7200000 BASH_MAX_TIMEOUT_MS=7200000 claude --permission-mode bypassPermissions -p "Follow the shepherd analyst skill. Your branch name is branch_name. <optional: context>" --output-format json | jq -r '"session_id=\(.session_id)\nmessage=\(.result)"'
  ```
  The `BASH_*_TIMEOUT_MS=7200000` prefix raises the analyst's Bash tool ceiling to 2h via process env (highest precedence), so the analyst can run the ~20–50 min eval as one blocking foreground call without being cut off at the ~10 min default. Keep this prefix on every Claude spawn.
- Your handoff prompt to agents must follow the exact given format. Do NOT give it any specific directions or suggestions. ONLY give it the branch name you setup for it and minimal context.
- Stay in one directory. Switching directories causes Codex to discover the session in a different namespace than the one it was originally spawned in, which causes analysis agents to fail to spawn. 
- Background agent completion is reported by task notifications. Wait for those notifications; use coarse polling only as a fallback. Agents can take a long time.

### Mangaging Large Agent Budgets
- It's expected that you might need to autonomously work for many days.
- Do NOT write a supervisor script. You are expected to directly manage and spawn analysis agents. Even if you are given a 1000 agent budget, you are expected to manage it over the course of days.

**Keep spawning new agents until the budget is exhausted.** Progress is always possible by spawning more serial agents and finding a better family of approach. ALWAYS create new branches for every agent.

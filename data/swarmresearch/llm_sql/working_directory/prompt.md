You are an expert in data optimization and LLM prompt caching. Your task is to
evolve the existing Evolved class to maximize prefix hit count (PHC) for efficient
LLM prompt caching.

Problem Context:
- You are given a pandas DataFrame with text data in rows and columns
- The goal is to reorder rows and choose a per-row column serialization order to maximize prefix reuse when processing rows sequentially
- Prefix reuse is measured on the serialized row strings induced by the returned row order and column_orderings
- This reduces LLM computation costs by reusing cached prefixes

Objective:
- Dual objective: (1) maximize prefix reuse across consecutive rows and (2) minimize runtime
- Combined score: 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12

Formally:
- For a given column ordering C, PHC(C) = sum over all rows r of hit(C, r)
- hit(C, r) = sum of len(df[r][C[f]])^2 for all f in prefix where df[r][C[f]] == df[r-1][C[f]];
  zero if mismatch starts at the first field.
- Runtime is measured as wall-clock seconds to compute the reordered DataFrame from the input.
- Combined score: 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12

Required API (DO NOT CHANGE):
- Keep the Evolved class structure and the reorder method signature:

    class Evolved(Algorithm):
        def reorder(
            self,
            df: pd.DataFrame,
            early_stop: int = 0,
            row_stop: int = None,
            col_stop: int = None,
            col_merge: List[List[str]] = [],
            one_way_dep: List[Tuple[str, str]] = [],
            distinct_value_threshold: float = 0.8,
            parallel: bool = True,
        ) -> Tuple[pd.DataFrame, List[List[str]]]:

- You can modify internal implementation but must preserve class structure and method signatures
- The reorder method must return a tuple of (reordered_dataframe, column_orderings)
- reordered_dataframe may reorder rows only; it must preserve the original schema and every row's cell values exactly
- column_orderings defines the serialization order for each returned row and is what the evaluator uses when scoring prefix reuse

Algorithm Design Guidelines:
- For each row, determine the optimal column order based on matches with the previous row
- Consider column statistics (unique values, string lengths) for ordering
- Focus on columns with high value frequency and long strings
- Handle missing values and mixed data types appropriately
- Optimize the existing recursive approach or replace it with more efficient vectorized methods
- Consider prefix-aware greedy approaches that condition on the current matched prefix

Constraints:
- You may reorder rows and choose different column orderings for different rows
- Do not add, remove, rename, merge, or physically reorder columns in the returned DataFrame
- Do not mutate any cell values; preserve every input row exactly, up to row order
- Return a DataFrame with the same shape as input
- Use exact string matching for prefix calculations
- Keep memory usage reasonable for large datasets
- Preserve all existing method signatures and class structure
- The algorithm will be called with the same parameters as the original Evolved

Evaluation:
- Run `./task-eval`
- The evaluation timeout is 360 seconds
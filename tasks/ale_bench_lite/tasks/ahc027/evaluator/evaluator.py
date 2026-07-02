import logging
import traceback
from pathlib import Path

from ale_bench.result import CaseResult, JudgeResult, Result
from safe_ale_session import start_ale_bench_session

logger = logging.getLogger(__name__ + "_ALE_BENCH_EVALUATOR")


def result_feedback(result: Result) -> CaseResult:
    if result.overall_judge_result == JudgeResult.ACCEPTED:
        return result.case_results[0]
    selected_case_idx = 0
    for idx, case_result in enumerate(result.case_results):
        if case_result.judge_result == result.overall_judge_result:
            selected_case_idx = idx
            break
    return result.case_results[selected_case_idx]


def evaluate(program_path):
    problem_id = "ahc027"
    logger.info("Evaluating program %s for %s", program_path, problem_id)
    try:
        session = start_ale_bench_session(problem_id=problem_id, lite_version=True, num_workers=13)
        if not session:
            raise RuntimeError("Failed to start or restart the session.")
        code = Path(program_path).read_text().replace("# EVOLVE-BLOCK-START", "").replace("# EVOLVE-BLOCK-END", "").strip()
        num_public_cases = 50
        cases = session.case_gen(list(range(num_public_cases)))
        public_result = session.case_eval(cases, code, code_language="cpp20", skip_local_visualization=True)
        extracted_case = result_feedback(public_result)
        combined_score = public_result.overall_absolute_score
        session.close()
        return {
            "judge_result": public_result.overall_judge_result.value,
            "overall_score": public_result.overall_absolute_score,
            "max_execution_time_sec": max(case_result.execution_time for case_result in public_result.case_results),
            "max_memory_usage_mib": max(case_result.memory_usage for case_result in public_result.case_results) // 1024 // 1024,
            "standard_error": extracted_case.error_str,
            "message": extracted_case.message,
            "combined_score": combined_score,
        }
    except Exception as exc:
        logger.error("Evaluation failed completely: %s", exc)
        logger.error(traceback.format_exc())
        return {"overall_score": 0.0, "error": str(exc)}


if __name__ == "__main__":
    from wrapper import run

    run(evaluate)

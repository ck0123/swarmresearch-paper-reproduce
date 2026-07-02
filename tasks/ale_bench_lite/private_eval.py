#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import ale_bench


def main(program_path: str, problem_id: str) -> dict:
    print(f"Problem ID: {problem_id}")
    print(f"Evaluating program: {program_path}")

    try:
        session = ale_bench.start(problem_id=problem_id, lite_version=False, num_workers=13)
        code = Path(program_path).read_text().replace("# EVOLVE-BLOCK-START", "").replace("# EVOLVE-BLOCK-END", "").strip()
        private_result, final_rank, final_performance = session.private_eval(code, code_language="cpp20")
        private_json = json.loads(private_result.model_dump_json(indent=4))

        private_passed_cases = 0
        private_failed_cases = 0
        for case in private_json["case_results"]:
            if case["judge_result"] == "ACCEPTED":
                private_passed_cases += 1
            else:
                private_failed_cases += 1

        metrics = {
            "private": {
                "private_rank": final_rank,
                "private_performance": final_performance,
                "private_score": private_result.overall_absolute_score,
                "num_private_passed_cases": private_passed_cases,
                "num_private_failed_cases": private_failed_cases,
            }
        }
        print(f"Current Resource Usage: {session.current_resource_usage}")
        print(f"Remaining Resources: {session.remaining_resource_usage}")
        return metrics
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "combined_score": 0.0,
            "public": {"judge_result": "REJECTED"},
            "private": {
                "private_rank": 0,
                "private_performance": 0,
                "private_score": 0,
                "num_private_passed_cases": 0,
                "num_private_failed_cases": 0,
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALE-Bench private evaluation")
    parser.add_argument("--program-path", type=str, default="program.cpp")
    parser.add_argument("--problem-id", type=str, default="ahc025")
    args = parser.parse_args()

    all_results = []
    for i in range(3):
        print(f"\n{'=' * 60}")
        print(f"Running evaluation {i + 1} of 3")
        print('=' * 60)
        all_results.append(main(args.program_path, args.problem_id))
        print('=' * 60)

    private_scores = [r["private"]["private_score"] for r in all_results]
    private_performances = [r["private"]["private_performance"] for r in all_results]
    private_ranks = [r["private"]["private_rank"] for r in all_results]
    passed_cases = [r["private"]["num_private_passed_cases"] for r in all_results]
    failed_cases = [r["private"]["num_private_failed_cases"] for r in all_results]

    summary = {
        "avg_private_score": sum(private_scores) / len(private_scores),
        "avg_private_performance": sum(private_performances) / len(private_performances),
        "avg_private_rank": sum(private_ranks) / len(private_ranks),
        "all_results": all_results,
        "avg_passed_cases": sum(passed_cases) / len(passed_cases),
        "avg_failed_cases": sum(failed_cases) / len(failed_cases),
    }
    print("\nFinal Summary:")
    print(json.dumps(summary, indent=2))

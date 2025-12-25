"""
Runner - оркестратор эволюционного цикла.

Цикл:
1. Запустить агента на задачах (все или выбранные)
2. Собрать результаты
3. Проанализировать провалы (Analyzer)
4. Сгенерировать улучшения (Evolver)
5. Сохранить новую версию (Versioner)
6. Проверить rollback условия
7. Повторить до target score или max iterations
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Загрузить env до использования
load_dotenv()  # Auto-find .env in current or parent directories

# Добавить parent dir в path для импорта agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from .versioner import Versioner
from .analyzer import Analyzer
from .evolver import Evolver


DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")


class EvolutionRunner:
    """Главный оркестратор эволюции."""

    def __init__(
        self,
        target_score: float = 80.0,
        max_iterations: int = 10,
        rollback_threshold: float = 0.1,
        analyzer_model: str = None,
        evolver_model: str = None,
        evolution_dir: str = "evolution",
        benchmark: str = "erc3-dev",
        workspace: str = "demo"
    ):
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.rollback_threshold = rollback_threshold
        self.benchmark = benchmark
        self.workspace = workspace

        # Используем модель из env или дефолт
        analysis_model = analyzer_model or DEFAULT_MODEL
        evolve_model = evolver_model or DEFAULT_MODEL

        self.versioner = Versioner(evolution_dir)
        self.analyzer = Analyzer(model=analysis_model)
        self.evolver = Evolver(model=evolve_model)

        self.evolution_dir = Path(evolution_dir)

    def run_full_evolution(self, agent_model: str = None) -> dict:
        """
        Запустить полный эволюционный цикл.

        Returns:
            dict с финальными результатами
        """
        print(f"\n{'='*70}")
        print("EVOLUTION CYCLE START")
        print(f"Target: {self.target_score} points")
        print(f"Max iterations: {self.max_iterations}")
        print(f"{'='*70}\n")

        best_score = 0.0
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            current_version = self.versioner.get_current_version()

            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{self.max_iterations} | Config v{current_version:03d}")
            print(f"{'='*70}\n")

            # 1. Запустить агента
            print("[1/5] Running agent on all tasks...")
            run_result = self._run_agent(agent_model)

            if not run_result["success"]:
                print(f"ERROR: Agent run failed: {run_result.get('error')}")
                break

            score = run_result["total_score"]
            tasks_passed = run_result["tasks_passed"]
            total_tasks = run_result["total_tasks"]

            print(f"\nScore: {score}/{total_tasks} ({score/total_tasks*100:.1f}%)")

            # Записать результат для текущей версии
            self.versioner.record_run_result(score, tasks_passed, total_tasks)

            # Проверить target
            if score >= self.target_score:
                print(f"\n🎉 TARGET REACHED! Score: {score}")
                return {
                    "success": True,
                    "final_score": score,
                    "final_version": current_version,
                    "iterations": iteration
                }

            # Проверить rollback
            if iteration > 1:
                history = self.versioner.get_history()
                if len(history) >= 2:
                    prev_score = history[-2]["score"]
                    if score < prev_score * (1 - self.rollback_threshold):
                        print(f"\n⚠️ Score degraded: {prev_score} → {score}")
                        print("Rolling back to previous version...")
                        self.versioner.rollback()
                        continue

            best_score = max(best_score, score)

            # 2. Анализировать провалы
            print("\n[2/5] Analyzing failures...")
            analysis_results = self._analyze_failures(run_result)

            if not analysis_results:
                print("No failures to analyze, but target not reached. Stopping.")
                break

            summary = self.analyzer.summarize_failures(analysis_results)
            print(f"Failed tasks: {summary['total_failed']}")
            if summary['patterns']:
                print("Top patterns:")
                for p in summary['patterns'][:3]:
                    print(f"  - {p['cause']}: {p['count']} tasks")

            # 3. Генерировать улучшения
            print("\n[3/5] Generating improvements...")
            current_config = self.versioner.get_current_config()
            evolution_result = self.evolver.evolve(current_config, analysis_results, summary)

            if not evolution_result.success:
                print(f"Evolution failed: {evolution_result.changes_description}")
                continue

            # 4. Сохранить новую версию
            print("\n[4/5] Saving new version...")
            new_version = self.versioner.save_new_version(
                new_config=evolution_result.new_config,
                analysis={"results": [r.to_dict() for r in analysis_results], "summary": summary},
                changes_description=evolution_result.changes_description,
                score=score,
                tasks_passed=tasks_passed,
                total=total_tasks
            )
            print(f"Created v{new_version:03d}")

            # 5. Показать изменения
            print("\n[5/5] Changes applied:")
            print(evolution_result.changes_description[:500])

        print(f"\n{'='*70}")
        print("EVOLUTION COMPLETE")
        print(f"Best score: {best_score}")
        print(f"Final version: v{self.versioner.get_current_version():03d}")
        print(f"{'='*70}\n")

        return {
            "success": best_score >= self.target_score,
            "final_score": best_score,
            "final_version": self.versioner.get_current_version(),
            "iterations": iteration
        }

    def run_focused(
        self,
        task_ids: list[str],
        max_iterations: int = 5,
        agent_model: str = None
    ) -> dict:
        """
        Запустить эволюцию на конкретных задачах (focused mode).

        Args:
            task_ids: Список task IDs для фокусированной эволюции
            max_iterations: Макс итераций для focused mode
            agent_model: Модель агента

        Returns:
            dict с результатами
        """
        print(f"\n{'='*70}")
        print(f"FOCUSED EVOLUTION: {len(task_ids)} tasks")
        print(f"Tasks: {', '.join(task_ids)}")
        print(f"{'='*70}\n")

        for iteration in range(max_iterations):
            current_version = self.versioner.get_current_version()
            print(f"\n--- Iteration {iteration+1}/{max_iterations} (v{current_version:03d}) ---\n")

            # Запустить агента только на выбранных задачах
            run_result = self._run_agent(agent_model, task_filter=task_ids)

            if not run_result["success"]:
                print(f"ERROR: {run_result.get('error')}")
                break

            passed = run_result["tasks_passed"]
            total = run_result["total_tasks"]
            print(f"Passed: {passed}/{total}")

            # Если все задачи прошли - успех
            if passed == total:
                print(f"\n✓ All focused tasks pass!")
                return {"success": True, "iterations": iteration + 1}

            # Анализировать провалы
            analysis_results = self._analyze_failures(run_result)
            if not analysis_results:
                break

            # Quick fix для каждой провальной задачи
            current_config = self.versioner.get_current_config()
            for result in analysis_results:
                if not result.passed:
                    print(f"Quick fix for {result.task_id}...")
                    fix_result = self.evolver.quick_fix(current_config, result)
                    if fix_result.success:
                        current_config = fix_result.new_config
                        print(f"  Applied: {fix_result.changes_description[:100]}")

            # Сохранить обновлённый config
            summary = self.analyzer.summarize_failures(analysis_results)
            new_version = self.versioner.save_new_version(
                new_config=current_config,
                analysis={"results": [r.to_dict() for r in analysis_results], "summary": summary},
                changes_description=f"Focused fixes for: {', '.join(task_ids)}",
                score=passed,
                tasks_passed=passed,
                total=total
            )
            print(f"Saved v{new_version:03d}")

        return {
            "success": False,
            "iterations": iteration + 1,
            "final_version": self.versioner.get_current_version()
        }

    def _run_agent(self, model: str = None, task_filter: list[str] = None) -> dict:
        """
        Запустить агента и собрать результаты.

        Args:
            model: Модель для агента
            task_filter: Если указан, запускать только эти task IDs

        Returns:
            dict с результатами прогона
        """
        try:
            # Импортируем agent здесь чтобы избежать circular imports
            from agent import ERC3Agent

            agent = ERC3Agent(
                model=model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101"),
                evolution_dir=str(self.evolution_dir)
            )

            # Запустить сессию
            if task_filter:
                results = self._run_filtered_session(agent, task_filter)
            else:
                results = agent.run_session(
                    benchmark=self.benchmark,
                    workspace=self.workspace,
                    name=f"Evolution v{self.versioner.get_current_version():03d}"
                )

            # Подсчитать статистику
            total_score = sum(r["score"] for r in results)
            tasks_passed = sum(1 for r in results if r["score"] >= 1.0)

            # Сохранить результаты в папку версии
            session_id = agent.logger.session_dir.name if agent.logger else "unknown"
            self.versioner.save_session_results(session_id, results)

            return {
                "success": True,
                "results": results,
                "total_score": total_score,
                "tasks_passed": tasks_passed,
                "total_tasks": len(results),
                "logs_dir": str(agent.logger.session_dir) if agent.logger else None
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _run_filtered_session(self, agent, task_filter: list[str]) -> list[dict]:
        """Запустить сессию только для выбранных задач."""
        from erc3 import ERC3

        core = ERC3()

        res = core.start_session(
            benchmark=self.benchmark,
            workspace=self.workspace,
            name=f"Focused Evolution v{self.versioner.get_current_version():03d}",
            architecture=f"Focused on: {', '.join(task_filter)}"
        )

        session_id = res.session_id
        agent.logger = agent.__class__.__bases__[0]  # Hack to avoid circular
        from agent import TaskLogger
        agent.logger = TaskLogger("output", session_id, agent.config_version)

        status = core.session_status(session_id)
        results = []

        for task in status.tasks:
            # Фильтруем по spec_id
            is_target = task.spec_id in task_filter

            core.start_task(task)

            if is_target:
                # Решаем целевую задачу
                agent.logger.start_task(task.spec_id, task.task_text)

                # Get API client
                from agent import CompanyAPIClient
                api_base = f"{core.base_url.rstrip('/')}/{self.benchmark}/{task.task_id}"
                api_client = CompanyAPIClient(api_base, {"Authorization": f"Bearer {core.key}"})

                result = agent.solve_task(task, api_client)
                eval_result = core.complete_task(task)

                score = eval_result.eval.score if eval_result.eval else 0.0
                eval_logs = eval_result.eval.logs if eval_result.eval else None

                agent.logger.end_task(score, result.get("summary", ""), eval_logs)

                results.append({
                    "task_id": task.task_id,
                    "spec_id": task.spec_id,
                    "score": score,
                    "eval_logs": eval_logs
                })
            else:
                # Пропускаем остальные задачи - просто завершаем их
                core.complete_task(task)

        core.submit_session(session_id)
        agent.logger.save_summary(results)

        return results

    def _analyze_failures(self, run_result: dict) -> list:
        """Проанализировать провалы из результатов прогона."""
        results = run_result.get("results", [])
        logs_dir = run_result.get("logs_dir")

        if not logs_dir:
            return []

        logs_path = Path(logs_dir)
        analysis_results = []

        for task_result in results:
            if task_result["score"] >= 1.0:
                continue

            spec_id = task_result["spec_id"]
            log_file = logs_path / f"{spec_id}.json"

            if not log_file.exists():
                continue

            with open(log_file, "r", encoding="utf-8") as f:
                task_log = json.load(f)

            # Преобразовать в формат для analyzer
            task_log_formatted = {
                "task_id": spec_id,
                "task_text": task_result.get("task_text", ""),
                "logs": task_log
            }

            # Найти task_text из логов
            for entry in task_log:
                if entry.get("type") == "task_start":
                    task_log_formatted["task_text"] = entry.get("task_text", "")
                    break

            eval_result = {
                "task_id": spec_id,
                "passed": False,
                "score": task_result["score"],
                "logs": task_result.get("eval_logs", "")
            }

            analysis = self.analyzer.analyze_task(task_log_formatted, eval_result)
            analysis_results.append(analysis)

        return analysis_results

    def run_sequential(
        self,
        max_iter_per_task: int = 5,
        agent_model: str = None,
        start_from: str = None
    ) -> dict:
        """
        Последовательная эволюция: идём по задачам, при провале
        итерируемся по конкретной задаче до успеха.

        Args:
            max_iter_per_task: Макс итераций на одну задачу
            agent_model: Модель агента
            start_from: Task ID с которого начать (пропускает предыдущие)

        Returns:
            dict с результатами
        """
        from erc3 import ERC3
        from agent import ERC3Agent, TaskLogger, CompanyAPIClient

        print(f"\n{'='*70}")
        print("SEQUENTIAL EVOLUTION MODE")
        print(f"Max iterations per task: {max_iter_per_task}")
        if start_from:
            print(f"Starting from: {start_from}")
        print(f"{'='*70}\n")

        core = ERC3()

        # Создать сессию для получения списка задач
        res = core.start_session(
            benchmark=self.benchmark,
            workspace=self.workspace,
            name=f"Sequential Evolution v{self.versioner.get_current_version():03d}",
            architecture="Sequential task-by-task evolution"
        )

        session_id = res.session_id
        status = core.session_status(session_id)
        tasks = status.tasks

        print(f"Session: {session_id}")
        print(f"Total tasks: {len(tasks)}\n")

        # Найти индекс начальной задачи
        start_idx = 0
        if start_from:
            for i, t in enumerate(tasks):
                if t.spec_id == start_from:
                    start_idx = i
                    print(f"Skipping first {start_idx} tasks, starting from {start_from}\n")
                    break
            else:
                print(f"WARNING: Task '{start_from}' not found, starting from beginning\n")

        passed_tasks = []
        failed_tasks = []

        for task_idx, task in enumerate(tasks, 1):
            # Пропустить задачи до start_from
            if task_idx - 1 < start_idx:
                print(f"[SKIP] Task {task_idx}/{len(tasks)}: {task.spec_id}")
                continue
            spec_id = task.spec_id
            print(f"\n{'='*70}")
            print(f"TASK {task_idx}/{len(tasks)}: {spec_id}")
            print(f"{'='*70}")

            task_passed = False

            for iteration in range(max_iter_per_task):
                current_version = self.versioner.get_current_version()
                print(f"\n--- Attempt {iteration+1}/{max_iter_per_task} (v{current_version:03d}) ---")

                # Создать агента с текущим config
                agent = ERC3Agent(
                    model=agent_model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101"),
                    evolution_dir=str(self.evolution_dir)
                )
                agent.logger = TaskLogger("output", f"seq_{session_id}", agent.config_version)

                # Запустить задачу
                core.start_task(task)
                agent.logger.start_task(spec_id, task.task_text)

                api_base = f"{core.base_url.rstrip('/')}/{self.benchmark}/{task.task_id}"
                api_client = CompanyAPIClient(api_base, {"Authorization": f"Bearer {core.key}"})

                result = agent.solve_task(task, api_client)
                eval_result = core.complete_task(task)

                score = eval_result.eval.score if eval_result.eval else 0.0
                eval_logs = eval_result.eval.logs if eval_result.eval else None

                agent.logger.end_task(score, result.get("summary", ""), eval_logs)

                print(f"Score: {score}")

                if score >= 1.0:
                    print(f"✓ Task {spec_id} PASSED!")
                    task_passed = True
                    passed_tasks.append(spec_id)
                    break

                # Провал - анализируем и эволюционируем
                print(f"✗ Failed, analyzing...")

                # Загрузить лог задачи
                log_file = agent.logger.session_dir / f"{spec_id}.json"
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        task_log = json.load(f)

                    task_log_formatted = {
                        "task_id": spec_id,
                        "task_text": task.task_text,
                        "logs": task_log
                    }

                    eval_result_dict = {
                        "task_id": spec_id,
                        "passed": False,
                        "score": score,
                        "logs": eval_logs or ""
                    }

                    # Анализ
                    analysis = self.analyzer.analyze_task(task_log_formatted, eval_result_dict)

                    if analysis.root_cause:
                        print(f"Root cause: {analysis.root_cause[:100]}...")

                    # Quick fix
                    current_config = self.versioner.get_current_config()
                    fix_result = self.evolver.quick_fix(current_config, analysis)

                    if fix_result.success:
                        summary = {"total_failed": 1, "patterns": [], "top_fixes": []}
                        new_version = self.versioner.save_new_version(
                            new_config=fix_result.new_config,
                            analysis={"results": [analysis.to_dict()], "summary": summary},
                            changes_description=fix_result.changes_description,
                            score=score,
                            tasks_passed=0,
                            total=1
                        )
                        print(f"Applied fix, saved v{new_version:03d}")
                    else:
                        print(f"No fix generated: {fix_result.changes_description}")

            if not task_passed:
                print(f"✗ Task {spec_id} FAILED after {max_iter_per_task} attempts")
                failed_tasks.append(spec_id)

        # Завершить сессию
        core.submit_session(session_id)

        # Итог
        print(f"\n{'='*70}")
        print("SEQUENTIAL EVOLUTION COMPLETE")
        print(f"{'='*70}")
        print(f"Passed: {len(passed_tasks)}/{len(tasks)}")
        print(f"Failed: {len(failed_tasks)}")
        if failed_tasks:
            print(f"Failed tasks: {', '.join(failed_tasks)}")
        print(f"Final version: v{self.versioner.get_current_version():03d}")

        return {
            "success": len(failed_tasks) == 0,
            "passed": len(passed_tasks),
            "failed": len(failed_tasks),
            "failed_tasks": failed_tasks,
            "final_version": self.versioner.get_current_version()
        }


def main():
    """CLI для запуска эволюции."""
    import argparse

    parser = argparse.ArgumentParser(description="Evolution Runner")
    parser.add_argument("--target", type=float, default=80.0, help="Target score")
    parser.add_argument("--max-iter", type=int, default=10, help="Max iterations")
    parser.add_argument("--focused", nargs="+", help="Task IDs for focused evolution")
    parser.add_argument("--sequential", action="store_true", help="Sequential mode: iterate per task until pass")
    parser.add_argument("--start-from", help="Task ID to start from in sequential mode")
    parser.add_argument("--workspace", default="demo", help="ERC3 workspace")
    parser.add_argument("--model", help="Agent model override")

    args = parser.parse_args()

    runner = EvolutionRunner(
        target_score=args.target,
        max_iterations=args.max_iter,
        workspace=args.workspace
    )

    if args.sequential:
        result = runner.run_sequential(
            max_iter_per_task=args.max_iter,
            agent_model=args.model,
            start_from=args.start_from
        )
    elif args.focused:
        result = runner.run_focused(args.focused, agent_model=args.model)
    else:
        result = runner.run_full_evolution(agent_model=args.model)

    print(f"\nFinal result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()

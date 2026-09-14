#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
OpenClaw Benchmark Adapter for AgentSysPerf
=========================================

Drives OpenClaw legal reasoning tasks through any AgentInvoker.

OpenClaw tests agent capabilities on:
- Legal research and case law analysis
- Statutory interpretation
- Legal reasoning and argumentation
- Citation accuracy
- Multi-jurisdictional legal understanding

Reference: https://github.com/your-org/openclaw (placeholder)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.protocols import (
    AgentInvoker,
    TaskResult,
    TaskSpec,
)

logger = logging.getLogger(__name__)


class OpenClawAdapter:
    """BenchmarkAdapter for OpenClaw legal reasoning benchmark.

    Parameters
    ----------
    dataset_path : Path, optional
        Path to OpenClaw dataset JSONL file. If None, generates sample tasks.
    case_law_dir : Path, optional
        Directory containing case law documents. If None, uses embedded samples.
    include_statutes : bool, default=True
        Whether to include statutory interpretation tasks.
    include_case_analysis : bool, default=True
        Whether to include case law analysis tasks.
    include_argumentation : bool, default=True
        Whether to include legal argumentation tasks.

    Examples
    --------
    >>> adapter = OpenClawAdapter(dataset_path=Path("openclaw_dataset.jsonl"))
    >>> tasks = list(adapter.list_tasks(limit=10))
    >>> len(tasks)
    10
    """

    name = "openclaw"
    version = "1.0.0"

    def __init__(
        self,
        *,
        dataset_path: Optional[Path] = None,
        case_law_dir: Optional[Path] = None,
        include_statutes: bool = True,
        include_case_analysis: bool = True,
        include_argumentation: bool = True,
    ) -> None:
        self._dataset_path = dataset_path
        self._case_law_dir = case_law_dir
        self._include_statutes = include_statutes
        self._include_case_analysis = include_case_analysis
        self._include_argumentation = include_argumentation
        self._loaded_tasks: Optional[list[dict]] = None

    # ─── BenchmarkAdapter Protocol ───────────────────────────────────

    def list_tasks(
        self,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[TaskSpec]:
        """Enumerate OpenClaw tasks.

        Parameters
        ----------
        include : list of str, optional
            Task IDs or categories to include (e.g., ["contract", "tort"]).
        exclude : list of str, optional
            Task IDs or categories to exclude.
        limit : int, optional
            Maximum number of tasks to return.

        Yields
        ------
        TaskSpec
            One legal reasoning task.
        """
        tasks = self._load_tasks()

        count = 0
        for task_data in tasks:
            # Filter by category
            if include and not any(
                task_data["id"].startswith(cat) or task_data["category"] == cat
                for cat in include
            ):
                continue

            if exclude and any(
                task_data["id"].startswith(cat) or task_data["category"] == cat
                for cat in exclude
            ):
                continue

            # Build TaskSpec
            yield TaskSpec(
                id=task_data["id"],
                instruction=self._format_instruction(task_data),
                category=task_data["category"],
                difficulty=task_data.get("difficulty", "medium"),
                timeout_s=task_data.get("timeout_s", 1800.0),  # 30 min default
                extra={
                    "question": task_data["question"],
                    "jurisdiction": task_data.get("jurisdiction", "US"),
                    "area_of_law": task_data.get("area_of_law", "general"),
                    "case_citations": task_data.get("case_citations", []),
                    "statutes": task_data.get("statutes", []),
                    "gold_answer": task_data.get("gold_answer", ""),
                    "gold_reasoning": task_data.get("gold_reasoning", ""),
                    "evaluation_criteria": task_data.get("evaluation_criteria", []),
                },
            )

            count += 1
            if limit and count >= limit:
                break

    def run_task(
        self,
        task: TaskSpec,
        *,
        agent_invoker: AgentInvoker,
        on_step: Optional[Any] = None,
    ) -> TaskResult:
        """Run one OpenClaw legal reasoning task.

        Parameters
        ----------
        task : TaskSpec
            The legal reasoning task to run.
        agent_invoker : AgentInvoker
            Handle to invoke the agent.
        on_step : callable, optional
            Callback for progress updates.

        Returns
        -------
        TaskResult
            Result with passed/failed verdict and detailed evaluation.
        """
        logger.info(f"Running OpenClaw task {task.id} ({task.category})")

        try:
            # Prepare legal context (case law + statutes)
            legal_context = self._prepare_legal_context(task)

            # Build full prompt with legal context
            prompt = self._build_prompt(task, legal_context)

            # Invoke agent (text-only task; no shell environment needed)
            agent_response = agent_invoker.invoke(
                instruction=prompt,
                metadata={
                    "task_type": "legal_reasoning",
                    "category": task.category,
                    "jurisdiction": task.extra["jurisdiction"],
                },
                session_hint=task.id,
            )

            # Extract answer from agent response
            agent_answer = self._extract_answer(agent_response)

            # Evaluate legal reasoning quality
            evaluation = self._evaluate_legal_reasoning(
                task=task,
                agent_answer=agent_answer,
                agent_response=agent_response,
            )

            logger.info(
                f"Task {task.id}: "
                f"score={evaluation['score']:.2f}, "
                f"passed={evaluation['passed']}"
            )

            return TaskResult(
                task_id=task.id,
                passed=evaluation["passed"],
                reward=evaluation["score"],
                error=None,
                extra={
                    "agent_answer": agent_answer,
                    "gold_answer": task.extra.get("gold_answer", ""),
                    "evaluation": evaluation,
                    "citations_used": evaluation.get("citations_used", []),
                    "reasoning_quality": evaluation.get("reasoning_quality", 0.0),
                    "citation_accuracy": evaluation.get("citation_accuracy", 0.0),
                },
            )

        except Exception as e:
            logger.error(f"Task {task.id} failed with error: {e}")
            return TaskResult(
                task_id=task.id,
                passed=False,
                reward=0.0,
                error=str(e),
            )

    def teardown(self) -> None:
        """Release resources (case law cache, etc.)."""
        logger.info("OpenClaw adapter teardown complete")
        self._loaded_tasks = None

    # ─── Internal Helpers ────────────────────────────────────────────

    def _load_tasks(self) -> list[dict]:
        """Load OpenClaw tasks from dataset or generate samples."""
        if self._loaded_tasks is not None:
            return self._loaded_tasks

        if self._dataset_path and self._dataset_path.exists():
            # Load from JSONL file
            tasks = []
            with open(self._dataset_path) as f:
                for line in f:
                    task = json.loads(line)
                    tasks.append(task)
            self._loaded_tasks = tasks
            logger.info(f"Loaded {len(tasks)} tasks from {self._dataset_path}")
        else:
            # Generate sample tasks for testing
            self._loaded_tasks = self._generate_sample_tasks()
            logger.warning(
                f"No dataset provided, using {len(self._loaded_tasks)} sample tasks"
            )

        return self._loaded_tasks

    def _generate_sample_tasks(self) -> list[dict]:
        """Generate sample OpenClaw tasks for testing."""
        samples = [
            {
                "id": "openclaw_contract_001",
                "category": "contract_law",
                "area_of_law": "breach_of_contract",
                "jurisdiction": "US",
                "difficulty": "medium",
                "question": (
                    "A software company contracted with a client to deliver a custom "
                    "application by June 1, 2024. The contract specified that 'time is "
                    "of the essence.' The company delivered the application on June 15, "
                    "2024, but the client had already contracted with another vendor. "
                    "Is the client obligated to accept the late delivery?"
                ),
                "case_citations": [
                    "Hadley v. Baxendale (1854)",
                    "UCC § 2-309",
                ],
                "statutes": ["Uniform Commercial Code § 2-309"],
                "gold_answer": (
                    "No, the client is not obligated to accept late delivery when 'time "
                    "is of the essence' is specified in the contract. This clause makes "
                    "timely performance a material term."
                ),
                "gold_reasoning": (
                    "When a contract specifies 'time is of the essence,' timely "
                    "performance becomes a condition precedent. Failure to meet the "
                    "deadline constitutes a material breach, excusing the non-breaching "
                    "party from performance."
                ),
                "evaluation_criteria": [
                    "Identifies 'time is of the essence' clause",
                    "Explains material breach doctrine",
                    "Cites relevant authority",
                    "Applies facts correctly",
                ],
            },
            {
                "id": "openclaw_tort_001",
                "category": "tort_law",
                "area_of_law": "negligence",
                "jurisdiction": "US",
                "difficulty": "hard",
                "question": (
                    "A surgeon performs an operation while intoxicated, causing injury "
                    "to the patient. The patient sues for negligence. The surgeon argues "
                    "that the injury would have occurred regardless of intoxication due "
                    "to an unexpected complication. Does the 'but-for' causation test "
                    "bar recovery?"
                ),
                "case_citations": [
                    "Palsgraf v. Long Island Railroad Co. (1928)",
                    "Summers v. Tice (1948)",
                ],
                "statutes": [],
                "gold_answer": (
                    "No, the substantial factor test may allow recovery even if 'but-for' "
                    "causation is uncertain. The surgeon's intoxication was a substantial "
                    "factor in creating risk, even if causation is difficult to prove."
                ),
                "gold_reasoning": (
                    "When multiple sufficient causes exist or causation is uncertain, "
                    "courts may use the substantial factor test instead of strict "
                    "'but-for' causation. The surgeon's conduct was wrongful and "
                    "materially increased risk of harm."
                ),
                "evaluation_criteria": [
                    "Recognizes causation difficulty",
                    "Discusses substantial factor test",
                    "Distinguishes from but-for causation",
                    "Addresses policy considerations",
                ],
            },
            {
                "id": "openclaw_constitutional_001",
                "category": "constitutional_law",
                "area_of_law": "first_amendment",
                "jurisdiction": "US",
                "difficulty": "hard",
                "question": (
                    "A state law requires social media companies to carry all user speech "
                    "without content moderation, claiming it protects free speech. Does "
                    "this law violate the First Amendment rights of the platforms?"
                ),
                "case_citations": [
                    "Miami Herald Publishing Co. v. Tornillo (1974)",
                    "Turner Broadcasting System, Inc. v. FCC (1994)",
                ],
                "statutes": ["First Amendment to the U.S. Constitution"],
                "gold_answer": (
                    "Yes, the law likely violates the platforms' First Amendment rights. "
                    "Content moderation is editorial discretion, and the government cannot "
                    "compel private entities to host speech against their will."
                ),
                "gold_reasoning": (
                    "Under Miami Herald, the government cannot force private entities to "
                    "carry speech. Social media platforms have editorial discretion as "
                    "publishers, and content moderation is protected editorial judgment."
                ),
                "evaluation_criteria": [
                    "Identifies state action vs. private actor",
                    "Applies Miami Herald precedent",
                    "Discusses editorial discretion",
                    "Addresses compelled speech doctrine",
                ],
            },
        ]
        return samples

    def _format_instruction(self, task_data: dict) -> str:
        """Format task as instruction for agent."""
        instruction = f"""Legal Reasoning Task ({task_data['category']})

Question: {task_data['question']}

Jurisdiction: {task_data.get('jurisdiction', 'US')}
Area of Law: {task_data.get('area_of_law', 'general')}

Provide a well-reasoned legal analysis that:
1. Identifies the relevant legal principles
2. Applies those principles to the facts
3. Cites relevant case law or statutes
4. Reaches a clear conclusion

Your answer should demonstrate legal reasoning, not just state a conclusion.
"""
        return instruction

    def _prepare_legal_context(self, task: TaskSpec) -> dict:
        """Prepare legal context (case law + statutes) for agent."""
        context = {
            "jurisdiction": task.extra["jurisdiction"],
            "area_of_law": task.extra["area_of_law"],
            "case_citations": task.extra.get("case_citations", []),
            "statutes": task.extra.get("statutes", []),
        }

        # In a full implementation, load actual case law documents
        # For now, provide citation references
        if self._case_law_dir and self._case_law_dir.exists():
            context["case_law_documents"] = self._load_case_law(
                task.extra.get("case_citations", [])
            )

        return context

    def _load_case_law(self, citations: list[str]) -> dict:
        """Load full case law documents (placeholder)."""
        # In production, this would load actual case documents
        # from a case law database or document repository
        case_docs = {}
        for citation in citations:
            # Placeholder: would load from self._case_law_dir
            case_docs[citation] = f"[Case law document for {citation}]"
        return case_docs

    def _build_prompt(self, task: TaskSpec, legal_context: dict) -> str:
        """Build full prompt with task instruction and legal context."""
        prompt_parts = [task.instruction]

        if legal_context.get("case_citations"):
            prompt_parts.append("\n\nRelevant Case Citations:")
            for citation in legal_context["case_citations"]:
                prompt_parts.append(f"- {citation}")

        if legal_context.get("statutes"):
            prompt_parts.append("\n\nRelevant Statutes:")
            for statute in legal_context["statutes"]:
                prompt_parts.append(f"- {statute}")

        return "\n".join(prompt_parts)

    def _extract_answer(self, agent_response: Any) -> str:
        """Extract legal answer from agent response."""
        if isinstance(agent_response, str):
            return agent_response
        elif isinstance(agent_response, dict):
            # Try common keys
            for key in ["answer", "output", "response", "result"]:
                if key in agent_response:
                    return str(agent_response[key])
            return str(agent_response)
        else:
            return str(agent_response)

    def _evaluate_legal_reasoning(
        self,
        task: TaskSpec,
        agent_answer: str,
        agent_response: Any,
    ) -> dict:
        """Evaluate legal reasoning quality.

        Scoring dimensions:
        - Correctness (does answer match gold standard)
        - Reasoning quality (logical structure, analysis depth)
        - Citation accuracy (correct use of case law/statutes)
        - Completeness (addresses all aspects of question)

        Returns
        -------
        dict with keys: score (0-1), passed (bool), breakdown
        """
        evaluation = {
            "score": 0.0,
            "passed": False,
            "reasoning_quality": 0.0,
            "citation_accuracy": 0.0,
            "completeness": 0.0,
            "citations_used": [],
        }

        agent_lower = agent_answer.lower()
        gold_answer = task.extra.get("gold_answer", "").lower()
        criteria = task.extra.get("evaluation_criteria", [])

        # 1. Correctness (40%) - Check if key concepts match gold answer
        correctness_score = self._check_correctness(agent_lower, gold_answer)

        # 2. Reasoning quality (30%) - Check for legal reasoning structure
        reasoning_score = self._check_reasoning_quality(
            agent_answer, criteria
        )

        # 3. Citation accuracy (20%) - Check if relevant cases/statutes cited
        citation_score, citations_used = self._check_citations(
            agent_answer, task.extra.get("case_citations", [])
        )

        # 4. Completeness (10%) - Check if evaluation criteria met
        completeness_score = self._check_completeness(agent_lower, criteria)

        # Weighted average
        total_score = (
            0.4 * correctness_score +
            0.3 * reasoning_score +
            0.2 * citation_score +
            0.1 * completeness_score
        )

        evaluation["score"] = total_score
        evaluation["passed"] = total_score >= 0.7  # 70% threshold
        evaluation["reasoning_quality"] = reasoning_score
        evaluation["citation_accuracy"] = citation_score
        evaluation["completeness"] = completeness_score
        evaluation["citations_used"] = citations_used

        return evaluation

    def _check_correctness(self, agent_answer: str, gold_answer: str) -> float:
        """Check if agent answer matches gold standard (simple keyword match)."""
        if not gold_answer:
            return 0.5  # No gold answer to compare

        # Extract key phrases from gold answer
        gold_words = set(gold_answer.split())
        agent_words = set(agent_answer.split())

        # Calculate overlap
        overlap = len(gold_words & agent_words)
        total = len(gold_words)

        if total == 0:
            return 0.5

        return min(1.0, overlap / total)

    def _check_reasoning_quality(self, agent_answer: str, criteria: list) -> float:
        """Check for legal reasoning structure and depth."""
        score = 0.0
        checks = 0

        # Check for reasoning structure
        reasoning_indicators = [
            "because", "therefore", "thus", "consequently",
            "under", "according to", "applying", "analysis"
        ]
        for indicator in reasoning_indicators:
            if indicator in agent_answer.lower():
                score += 0.2
                checks += 1
                if checks >= 3:
                    break

        # Check for criteria coverage
        if criteria:
            criteria_met = sum(
                1 for criterion in criteria
                if any(word in agent_answer.lower() for word in criterion.lower().split())
            )
            score += (criteria_met / len(criteria)) * 0.4

        return min(1.0, score)

    def _check_citations(self, agent_answer: str, expected_citations: list) -> tuple:
        """Check if relevant citations are used correctly."""
        citations_used = []

        for citation in expected_citations:
            # Extract case name (before parenthesis)
            case_name = citation.split("(")[0].strip()
            if case_name.lower() in agent_answer.lower():
                citations_used.append(citation)

        if not expected_citations:
            return 0.5, []  # No expected citations

        citation_rate = len(citations_used) / len(expected_citations)
        return citation_rate, citations_used

    def _check_completeness(self, agent_answer: str, criteria: list) -> float:
        """Check if all evaluation criteria are addressed."""
        if not criteria:
            return 0.5  # No criteria to check

        criteria_met = 0
        for criterion in criteria:
            # Simple keyword check
            if any(word.lower() in agent_answer for word in criterion.split()):
                criteria_met += 1

        return criteria_met / len(criteria)


__all__ = ["OpenClawAdapter"]

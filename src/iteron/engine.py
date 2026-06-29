import difflib
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import cognitive
from . import memory as mem
from . import models
from .safety import EvalContract, Chronicle, atomic_write

EXPERIMENTS_DIR = Path(os.environ.get("ITERON_EXPERIMENTS", "experiments"))


class Iteron:
    def __init__(self, name: str):
        self.name = name
        self.exp_dir = EXPERIMENTS_DIR / name
        self.config = self._load_config()
        self.state = self._load_state()
        self.journal_path = self.exp_dir / "journal.jsonl"
        self.dmf = mem.DMF(self.exp_dir / "archive")
        self.contract = EvalContract(self.exp_dir)
        self.chronicle = Chronicle(self.exp_dir)
        self.model = models.ModelClient(
            provider=self.config.get("model", {}).get("provider")
        )

    def _load_config(self) -> dict:
        import yaml
        path = self.exp_dir / "config.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            return yaml.safe_load(f)

    def _load_state(self) -> dict:
        path = self.exp_dir / "state.json"
        if path.exists():
            return json.loads(path.read_text())
        return {
            "phase": "cold_start",
            "round": -3,
            "best_score": 0.0,
            "best_round": -1,
            "total_cost": 0.0,
            "cpg": 0.0,
            "budget_remaining": 50.0,
            "last_scores": [],
            "per_round_gains": [],
            "meta_suggestions": [],
            "skills_prefix": "",
        }

    def _save_state(self):
        atomic_write(self.exp_dir / "state.json", json.dumps(self.state, indent=2))

    def _journal(self, entry: dict):
        entry["t"] = time.time()
        entry["round"] = self.state.get("round", -3)
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _add_cost(self, amount: float):
        self.state["total_cost"] += amount
        self.state["budget_remaining"] -= amount

    def _run_eval(self, solution_dir: Path) -> dict:
        self.contract.verify()
        self.chronicle.snapshot("pre_eval")  # ponytail: full-dir copy; incremental if >100MB
        try:
            result = subprocess.run(
                ["bash", str(self.contract.contract_path), str(solution_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return {
                    "score": 0.0,
                    "cost": 0.0,
                    "error": result.stderr.strip() or "non-zero exit",
                }
            parsed = json.loads(result.stdout.strip())
            if isinstance(parsed, (int, float)):
                parsed = {"score": float(parsed)}
            parsed.setdefault("score", 0.0)
            parsed.setdefault("cost", 0.0)
            return parsed
        except json.JSONDecodeError:
            return {"score": 0.0, "cost": 0.0, "error": "invalid JSON output"}
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "cost": 0.0, "error": "timeout"}
        except Exception as e:
            return {"score": 0.0, "cost": 0.0, "error": str(e)}
        finally:
            self.chronicle.restore("pre_eval")

    def _resolve_best(self) -> Optional[Path]:
        link = self.exp_dir / "best_solution"
        if link.is_symlink():
            try:
                target = link.resolve()
                return target if target.exists() else None
            except (OSError, RuntimeError):
                return None
        if link.is_dir() and (link / "solution.py").is_file():
            return link
        return None

    def _edit_signature(self, old: str, new: str) -> str:
        return hashlib.sha256(f"{old}\x00{new}".encode()).hexdigest()[:16]

    def _error_signature(self, error: str) -> str:
        return hashlib.sha256(error.encode()).hexdigest()[:16]

    def _extract_code(self, response: str) -> str:
        for m in ("```python\n", "```python ", "```\n", "``` "):
            if m in response:
                _, rest = response.split(m, 1)
                code = rest.rsplit("```", 1)[0].strip()
                if code:
                    return code
        return response.strip()

    def _compute_sdds_tag(self, old_code: str, new_code: str) -> str:
        # ponytail: full diff every round; hash-cache if >100 rounds/round
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        diff = list(difflib.unified_diff(old_lines, new_lines))
        changed = sum(1 for l in diff if l.startswith("+") or l.startswith("-"))
        n = len(old_lines) or 1
        if changed > 10 or changed / n > 0.2:
            return "H-space"
        return "E-space"

    def _detect_anomaly(self, score: float, recent: list[float]) -> bool:
        # ponytail: 2σ z-score; configurable threshold if false positives
        if len(recent) < 3:
            return False
        mean = sum(recent) / len(recent)
        var = sum((s - mean) ** 2 for s in recent) / len(recent)
        std = var ** 0.5
        if std == 0:
            return abs(score - mean) > 0.001
        return abs(score - mean) > 2 * std

    def _run_anomaly_debate(self, score: float, recent: list[float]) -> str:
        mean = sum(recent) / len(recent) if recent else 0.0
        a_prompt, b_prompt = cognitive.build_anomaly_debate_prompts(
            score, mean, self.config.get("problem", "")
        )
        a_resp = self.model.call(a_prompt, tier="fast", temperature=0.3)
        self._add_cost(0.002)
        b_resp = self.model.call(b_prompt, tier="fast", temperature=0.3)
        self._add_cost(0.002)
        # ponytail: heuristic — 60-char prefix guard against "genuine noise"
        a_gen = "genuine" in a_resp.lower() and "noise" not in a_resp.lower()[:60]
        b_noise = "noise" in b_resp.lower() and "genuine" not in b_resp.lower()[:60]
        if a_gen and not b_noise:
            return "genuine"
        if b_noise and not a_gen:
            return "noise"
        return "disagreement"

    def _escalate_to_problematizer(self):
        self._journal({"action": "escalation", "to": "problematizer"})
        problem = self.config.get("problem", "")
        current_code = ""
        best_dir = self._resolve_best()
        if best_dir:
            sol_path = best_dir / "solution.py"
            if sol_path.exists():
                current_code = sol_path.read_text()
        prompt = cognitive.build_problematizer_prompt(problem, current_code)
        try:
            response = self.model.call(prompt, tier="smart")
            self._add_cost(0.075)
            self._journal({
                "action": "problematizer_output",
                "output": response[:1000],
            })
        except models.ModelError as e:
            self._journal({"action": "problematizer_failed", "error": str(e)})

    def _build_self_model(self) -> dict:
        scores = self.state.get("last_scores", [])
        gains = self.state.get("per_round_gains", [])
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        n_sol = len(self.dmf.solution)
        n_ref = len(self.dmf.refinement)
        ref_confs = [v.get("confidence", 0) for v in self.dmf.refinement.all().values()]
        avg_conf = sum(ref_confs) / len(ref_confs) if ref_confs else 0.0

        return {
            "capabilities": {
                "agents": ["proposer", "evaluator", "debater", "problematizer"],
                "memory_systems": ["dmf_solution", "dmf_refinement", "dmf_execution"],
                "search_strategies": ["greedy", "cold_start"],
            },
            "performance": {
                "avg_score_improvement": round(avg_gain, 4),
                "cost_per_round": round(self.state.get("total_cost", 0) / max(self.state.get("round", 1), 1), 4),
                "cpg": round(self.state.get("cpg", 0), 2),
                "best_score": self.state.get("best_score", 0),
                "rounds_completed": self.state.get("round", 0),
                "solutions_in_memory": n_sol,
                "refinements_tracked": n_ref,
                "avg_refinement_confidence": round(avg_conf, 2),
                "errors_cached": len(self.dmf.execution),
            },
            "limitations": [],
            "improvement_hypotheses": [],
            "metacognitive_habits": {
                "cognitive_flexibility": "Maintain 2+ competing hypotheses via SDDS tagging",
                "epistemic_humility": "Treat best results as provisional — anomaly debate validates surprises",
                "bias_awareness": "Separate proposer from evaluator to reduce confirmation bias",
            },
        }

    def _meta_agent_suggest(self) -> Optional[str]:
        self_model = self._build_self_model()
        prompt = cognitive.build_meta_agent_prompt(
            json.dumps(self_model, indent=2)
        )
        try:
            response = self.model.call(prompt, tier="smart")
            self._add_cost(0.075)
            self._journal({
                "action": "meta_suggestion",
                "output": response[:1000],
            })
            summary = ""
            for line in response.split("\n"):
                if line.startswith("SUMMARY:"):
                    summary = line[len("SUMMARY:"):].strip()
                    break
            suggestion = {
                "response": response[:500],
                "summary": summary or response[:200],
            }
            suggestions = self.state.setdefault("meta_suggestions", [])
            suggestions.append(suggestion)
            self._save_state()
            return summary or response[:200]
        except models.ModelError as e:
            self._journal({"action": "meta_suggestion_failed", "error": str(e)})
            return None

    def _compress_skills(self):
        prompt = cognitive.build_skill_compression_prompt(
            self.dmf.solution.all(),
            self.dmf.refinement.all(),
            self.dmf.execution.all(),
        )
        try:
            response = self.model.call(prompt, tier="fast")
            self._add_cost(0.002)
            self.state["skills_prefix"] = response[:700]
            self._save_state()
        except models.ModelError as e:
            self._journal({"action": "compress_skills_failed", "error": str(e)})

    def _cold_start(self):
        if self.state.get("phase") == "greedy":
            return

        best_link = self.exp_dir / "best_solution"
        best_dir = None
        if best_link.is_symlink():
            try:
                t = best_link.resolve()
                if t.exists():
                    best_dir = t
            except (OSError, RuntimeError):
                pass
        if best_dir is None and best_link.is_dir() and (best_link / "solution.py").is_file():
            best_dir = best_link
        if best_dir is not None:
            return self._seed_from(best_dir)

        self.state["phase"] = "cold_start"
        problem = self.config.get("problem", "")
        if not problem:
            raise ValueError("config.yaml must contain a 'problem' field")

        drafts = []
        for i in range(3):
            try:
                prompt = cognitive.build_cold_start_prompt(problem)
                code = self.model.call(prompt, tier="fast", temperature=0.9)
            except models.ModelError as e:
                self._journal({"action": "cold_start_failed", "error": str(e)})
                continue
            draft_dir = self.exp_dir / f"draft_{i}"
            draft_dir.mkdir(exist_ok=True)
            (draft_dir / "solution.py").write_text(code)
            drafts.append((i, draft_dir))
            self.state["round"] = -3 + i
            self._add_cost(0.002)
            self._journal({
                "action": "cold_start_draft",
                "tag": f"draft_{i + 1}",
                "cost": 0.002,
            })
            self._save_state()

        if not drafts:
            raise RuntimeError("Cold start failed: no drafts generated")

        best_score = -1.0
        best_idx = -1
        for i, draft_dir in drafts:
            result = self._run_eval(draft_dir)
            score = result.get("score", 0.0)
            cost = result.get("cost", 0.0)
            self._add_cost(cost)
            self._journal({
                "action": "cold_start_eval",
                "draft": i + 1,
                "score": score,
                "cost": cost,
            })
            if score > best_score:
                best_score = score
                best_idx = i

        best_link = self.exp_dir / "best_solution"
        if best_link.is_dir():
            shutil.rmtree(best_link)
        best_link.unlink(missing_ok=True)
        if best_idx >= 0:
            best_dir = self.exp_dir / f"draft_{best_idx}"
            best_link.symlink_to(os.path.relpath(best_dir, self.exp_dir))
            self.state["best_score"] = best_score
            self.state["best_round"] = 0
            self.state["last_scores"] = [best_score]
            self.dmf.solution.put("round_0", {
                "score": best_score,
                "round": 0,
                "phase": "cold_start",
                "tag": f"draft_{best_idx + 1}",
            })

        for i in range(3):
            if i != best_idx:
                shutil.rmtree(self.exp_dir / f"draft_{i}", ignore_errors=True)

        if best_idx >= 0:
            self.state["phase"] = "greedy"
            self.state["round"] = 1
        else:
            self.state["phase"] = "failed"
        self._save_state()

    def _seed_from(self, seed_dir: Path):
        self._journal({"action": "seed_detected", "seed_dir": str(seed_dir)})
        result = self._run_eval(seed_dir)
        score = result.get("score", 0.0)
        round_n = 0
        self.state["round"] = round_n
        self.state["best_score"] = score
        self.state["best_round"] = round_n
        self.state["last_scores"] = [score]
        self.state["phase"] = "greedy"
        self.dmf.solution.put(f"round_{round_n}", {
            "score": score, "round": round_n, "phase": "seed",
        })
        self._add_cost(result.get("cost", 0.0))
        best_link = self.exp_dir / "best_solution"
        if best_link != seed_dir:
            if best_link.is_dir():
                shutil.rmtree(best_link)
            best_link.unlink(missing_ok=True)
            best_link.symlink_to(os.path.relpath(seed_dir, self.exp_dir))
        self._save_state()

    def _greedy_loop(self):
        scores = list(self.state.get("last_scores", []))
        gains = list(self.state.get("per_round_gains", []))
        window = self.config.get("plateau_window", 5)
        threshold = self.config.get("plateau_threshold", 0.01)

        while self.state["budget_remaining"] > 0:
            if self.state.get("phase") == "problematizer":
                break

            best_dir = self._resolve_best()
            if not best_dir:
                break
            sol_path = best_dir / "solution.py"
            if not sol_path.exists():
                self._journal({"action": "propose_skip", "reason": "no solution.py"})
                continue
            current_code = sol_path.read_text()
            dmf_context = self._get_dmf_context()
            sys_prompt, user_prompt = cognitive.build_proposal_prompt(
                self.config.get("problem", ""), current_code, dmf_context
            )

            try:
                new_code = self.model.call(
                    user_prompt, system=sys_prompt, tier="fast"
                )
            except models.ModelError as e:
                self._journal({"action": "propose_failed", "error": str(e)})
                self.state["round"] += 1
                self._save_state()
                continue
            llm_cost = self.model.estimate_cost(new_code, "fast")
            self._add_cost(llm_cost)

            sdds_tag = self._compute_sdds_tag(current_code, new_code)

            candidate_dir = self.exp_dir / "candidate"
            shutil.rmtree(candidate_dir, ignore_errors=True)
            candidate_dir.mkdir()
            (candidate_dir / "solution.py").write_text(self._extract_code(new_code))

            result = self._run_eval(candidate_dir)
            score = result.get("score", 0.0)
            eval_cost = result.get("cost", 0.0)
            self._add_cost(eval_cost)

            # --- Anomaly detection (Stage 1 + 2) ---
            if self._detect_anomaly(score, scores):
                self._journal({
                    "action": "reality_check",
                    "reason": f"score {score:.4f} vs recent mean",
                })
                result2 = self._run_eval(candidate_dir)
                score2 = result2.get("score", 0.0)
                self._add_cost(result2.get("cost", 0.0))
                if self._detect_anomaly(score2, scores):
                    debate_result = self._run_anomaly_debate(score2, scores)
                    self._journal({
                        "action": "anomaly_debate",
                        "result": debate_result,
                        "score": score2,
                    })
                    if debate_result == "genuine":
                        self._escalate_to_problematizer()
                result = result2
                score = score2

            old_best = self.state["best_score"]
            improved = score > old_best and not result.get("error")

            round_key = f"round_{self.state['round']}"
            self.dmf.solution.put(round_key, {
                "score": score,
                "round": self.state["round"],
                "phase": "greedy",
                "tag": sdds_tag,
            })

            edit_key = self._edit_signature(current_code, new_code)
            delta = score - old_best
            existing_ref = self.dmf.refinement.get(edit_key)
            if existing_ref:
                c = existing_ref["count"] + 1
                existing_ref["delta"] = (
                    existing_ref["delta"] * existing_ref["count"] + delta
                ) / c
                existing_ref["confidence"] = (
                    existing_ref["confidence"] * 0.95
                    + (0.05 if improved else 0)
                )
                existing_ref["count"] = c
                self.dmf.refinement.put(edit_key, existing_ref)
            else:
                self.dmf.refinement.put(edit_key, {
                    "delta": delta,
                    "confidence": 0.5,
                    "count": 1,
                })

            if result.get("error"):
                err_key = self._error_signature(result["error"])
                existing_err = self.dmf.execution.get(err_key)
                if existing_err:
                    existing_err["count"] += 1
                    self.dmf.execution.put(err_key, existing_err)
                else:
                    self.dmf.execution.put(err_key, {
                        "error": result["error"][:200],
                        "fix": new_code[:500],
                        "count": 1,
                        "verified": False,
                    })

            if improved:
                new_best = self.exp_dir / f"round_{self.state['round']}"
                shutil.rmtree(new_best, ignore_errors=True)
                shutil.copytree(candidate_dir, new_best)
                best_link = self.exp_dir / "best_solution"
                if best_link.is_dir():
                    shutil.rmtree(best_link)
                best_link.unlink(missing_ok=True)
                best_link.symlink_to(
                    os.path.relpath(new_best, self.exp_dir)
                )
                self.state["best_score"] = score
                self.state["best_round"] = self.state["round"]
                self._git_commit(score)

            shutil.rmtree(candidate_dir, ignore_errors=True)
            scores.append(score)
            gains.append(max(0.0, score - old_best) if improved else 0.0)
            if len(scores) > window:
                scores.pop(0)
            if len(gains) > window:
                gains.pop(0)

            self._journal({
                "action": "propose",
                "score": score,
                "cost": round(llm_cost + eval_cost, 6),
                "improved": improved,
                "phase": "greedy",
                "tag": sdds_tag,
            })
            self.state["round"] += 1

            total_cost_window = sum(
                e.get("cost", 0) for e in self._recent_journal(window)
            )
            total_gain_window = sum(gains) or 0.001
            self.state["cpg"] = total_cost_window / total_gain_window

            self.state["last_scores"] = scores[-window:]
            self.state["per_round_gains"] = gains[-window:]
            self._save_state()

            if self._detect_plateau(scores, window, threshold):
                self._journal({
                    "action": "plateau_detected",
                    "scores": scores[-window:],
                })
                self._escalate_to_problematizer()
                # Phase 4: Meta Agent runs after problematizer
                if self.state["budget_remaining"] > 0:
                    self._meta_agent_suggest()
                # Phase 5: compress DMF into skills prefix
                if self.state["budget_remaining"] > 0:
                    self._compress_skills()
                break

    def _recent_journal(self, n: int) -> list[dict]:
        # ponytail: O(n) scan per round; rolling window cost if rounds > 100
        entries = []
        try:
            with open(self.journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return entries[-n:]

    def _detect_plateau(
        self, scores: list[float], window: int, threshold: float
    ) -> bool:
        if len(scores) < window:
            return False
        return max(scores[-window:]) - min(scores[-window:]) < threshold

    def _get_dmf_context(self) -> str:
        parts = []
        if len(self.dmf.solution) > 0:
            best_key = max(
                self.dmf.solution.keys(),
                key=lambda k: self.dmf.solution.get(k, {}).get("score", 0),
            )
            best = self.dmf.solution.get(best_key)
            if best:
                parts.append(f"Best known score in memory: {best['score']}")
        if len(self.dmf.refinement) > 0:
            effective = [
                v for v in self.dmf.refinement.all().values()
                if v.get("confidence", 0) > 0.6
            ]
            if effective:
                top = max(effective, key=lambda x: x["confidence"])
                parts.append(
                    f"Known effective refinement: "
                    f"delta={top['delta']:.4f} "
                    f"(confidence={top['confidence']:.2f})"
                )
        if len(self.dmf.execution) > 0:
            parts.append(f"{len(self.dmf.execution)} known errors in cache.")
        suggestions = self.state.get("meta_suggestions", [])
        if suggestions:
            last = suggestions[-1].get("summary", "")
            if last:
                parts.append(f"Last meta suggestion: {last}")
        skills = self.state.get("skills_prefix", "")
        if skills:
            parts.append(f"Internalized skills:\n{skills}")
        return "\n".join(parts)

    def _git_commit(self, score: float):
        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.exp_dir,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git", "commit", "-m",
                    f"round {self.state['round']}: score {score:.4f}",
                    "--allow-empty",
                ],
                cwd=self.exp_dir,
                capture_output=True,
            )
        except Exception as e:
            self._journal({"action": "git_commit_failed", "error": str(e)})

    def run(self):
        self.state["budget_remaining"] = self.config.get("budget", 50.0)
        self._cold_start()
        while self.state.get("phase") != "failed" and self.state["budget_remaining"] > 0:
            self._greedy_loop()

    def status(self) -> dict:
        s = {
            "experiment": self.name,
            "phase": self.state.get("phase"),
            "round": self.state.get("round"),
            "best_score": self.state.get("best_score"),
            "best_round": self.state.get("best_round"),
            "total_cost": round(self.state.get("total_cost", 0), 4),
            "budget_remaining": round(
                self.state.get("budget_remaining", 0), 4
            ),
            "cpg": round(self.state.get("cpg", 0), 4),
            "dmf": {
                "solutions": len(self.dmf.solution),
                "refinements": len(self.dmf.refinement),
                "errors_cached": len(self.dmf.execution),
            },
            "meta_suggestions": len(self.state.get("meta_suggestions", [])),
            "skills_prefix_len": len(self.state.get("skills_prefix", "")),
        }
        return s

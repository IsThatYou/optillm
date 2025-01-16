import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class PlanSearch:
    def __init__(self, system_prompt: str, client, model: str):
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.token_counts = {'prompt_tokens': 0, 'completion_tokens': 0}

    def generate_observations(self, problem: str, num_observations: int = 3) -> List[str]:
        prompt = f"""You are an expert problem solver. You will be given a problem to analyze.
You will return several useful, non-obvious, and insightful observations about the problem that could help solve it.
Focus on identifying key patterns, constraints, and hidden relationships that might not be immediately apparent.
Be creative and think beyond conventional approaches.

Here is the problem:
{problem}

Please provide {num_observations} key observations."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.token_counts['prompt_tokens'] += response.usage.prompt_tokens
        self.token_counts['completion_tokens'] += response.usage.completion_tokens
        observations = response.choices[0].message.content.strip().split('\n')
        return [obs.strip() for obs in observations if obs.strip()]

    def generate_derived_observations(self, problem: str, observations: List[str], num_new_observations: int = 2) -> List[str]:
        prompt = f"""You are an expert problem solver. You will be given a problem and several insightful observations about it.
You will brainstorm new observations by combining and extending the existing ones in creative ways.
Look for connections between observations and potential implications that weren't initially obvious.

Here is the problem:
{problem}

Here are the existing observations:
{chr(10).join(f"{i+1}. {obs}" for i, obs in enumerate(observations))}

Please provide {num_new_observations} new observations derived from the existing ones."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.token_counts['prompt_tokens'] += response.usage.prompt_tokens
        self.token_counts['completion_tokens'] += response.usage.completion_tokens
        new_observations = response.choices[0].message.content.strip().split('\n')
        return [obs.strip() for obs in new_observations if obs.strip()]

    def generate_solution(self, problem: str, observations: List[str]) -> str:
        prompt = f"""Here is the problem to solve:
{problem}

Here are the key insights to help solve the problem:
{chr(10).join(f"Insight {i+1}: {obs}" for i, obs in enumerate(observations))}

Use these insights above to develop a clear, step-by-step solution to the problem.
Think creatively and go beyond conventional approaches, while ensuring your solution is logical and well-supported.
Quote relevant insights EXACTLY before each step to show how they inform your solution. QUOTING IS CRUCIAL."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.token_counts['prompt_tokens'] += response.usage.prompt_tokens
        self.token_counts['completion_tokens'] += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def implement_solution(self, problem: str, solution: str) -> str:
        prompt = f"""You are an expert problem solver. You will be given a problem and a detailed solution approach.
Transform this solution into a concrete implementation that solves the problem.
Your implementation should closely follow the solution's logic while being efficient and practical.

Problem:
{problem}

Solution Approach:
{solution}

Please implement the solution."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        self.token_counts['prompt_tokens'] += response.usage.prompt_tokens
        self.token_counts['completion_tokens'] += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def solve(self, problem: str, num_initial_observations: int = 3, num_derived_observations: int = 2) -> Tuple[str, str]:
        logger.info("Generating initial observations")
        initial_observations = self.generate_observations(problem, num_initial_observations)
        
        logger.info("Generating derived observations")
        derived_observations = self.generate_derived_observations(problem, initial_observations, num_derived_observations)
        
        all_observations = initial_observations + derived_observations
        
        logger.info("Generating solution based on observations")
        natural_language_solution = self.generate_solution(problem, all_observations)
        
        logger.info("Implementing solution in Python")
        python_implementation = self.implement_solution(problem, natural_language_solution)
        
        return natural_language_solution, python_implementation

    def solve_multiple(self, problem: str, n: int, num_initial_observations: int = 3, num_derived_observations: int = 2) -> List[str]:
        solutions = []
        for _ in range(n):
            _, python_implementation = self.solve(problem, num_initial_observations, num_derived_observations)
            solutions.append(python_implementation)
        return solutions

def plansearch(system_prompt: str, initial_query: str, client, model: str, n: int = 1) -> List[str]:
    planner = PlanSearch(system_prompt, client, model)
    return planner.solve_multiple(initial_query, n), planner.token_counts

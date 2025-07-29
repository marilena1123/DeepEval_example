import json
import os
import argparse
from dotenv import load_dotenv

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
import litellm

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

class LiteLLMWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_key: str, api_base: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.model = self.load_model()

    def get_model_name(self):
        return self.model_name

    def load_model(self):
        litellm.api_key = self.api_key
        if self.api_base:
            litellm.api_base = self.api_base
        return litellm

    def generate(self, prompt: str) -> str:
        response = self.model.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()

def main(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    llm = LiteLLMWrapper(
        model_name="openai/gpt-4o",
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_BASE_URL
    )

    metric = GEval(
        name="Math Correctness",
        criteria="Is the answer mathematically correct for the given problem? Ignore wording. Just focus on accuracy and give a short reason.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=llm,
        verbose_mode=True
    )

    results = []
    for item in data:
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["model_answer"],
            expected_output=item["answer"]
        )
        metric.measure(test_case)

        results.append({
            "question": item["question"],
            "expected_answer": item["answer"],
            "model_answer": item["model_answer"],
            "evaluation": {
                "score": metric.score,
                "reason": metric.reason
            }
        })

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate math answer correctness using GEval.")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    main(args.input, args.output)

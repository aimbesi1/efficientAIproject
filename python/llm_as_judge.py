from typing import Literal
import pandas as pd
import time

# Set environment variable:
# set OPENAI_API_KEY=XXXX

from mirascope.core import openai, prompt_template
from pydantic import BaseModel, Field

class Eval(BaseModel):
    score: Literal["poor", "ok", "good", "great", "excellent"] = Field(
        ..., description="A score representing the relevance of the generated answer."
    )
    reasoning: str = Field(
        ..., description="The reasoning for the score in 100 characters or less."
    )

@openai.call(model="gpt-4o-mini", response_model=Eval)
@prompt_template(
    """
    Evaluate the relevance of the generated response to the given writing prompt on a continuous scale from poor to excellent.
    A generation is considered relevant (score > poor ) if it addresses the core demands of the response, follows any requested style, 
    and stays focused on the topic.

    Consider the following aspects when evaluating the response:
    - Style: Does the response follow the style or tone requested in the prompt?
    - Completeness: Does the response sufficiently cover the prompt?
    - Focus: Does the response stay on topic without introducing unnecessary or unrelated information?
    - Clarity: Is the response clear and easy to understand?
    - Accuracy: If the prompt requires factual information, is the response factually correct?

    Use the following relevance scale:
    
    poor - No relevance; the answer is completely unrelated or nonsensical
    ok - Low relevance; minor relation to the prompt but missing key details or accuracy
    good - Moderate relevance; somewhat addresses the prompt but lacks depth or focus
    great - High relevance; mostly follows the prompt but may lack completeness or introduce some off-topic information
    excellent - Very high relevance; thoroughly follows the prompt with minor flaws

    Provide a brief reasoning for your assigned score, considering different aspects such as clarity, focus, and completeness.

    Prompt: {prompt}
    Response: {response}
    """
)
def evaluate_answer_relevance(prompt: str, response: str): ...
df = pd.read_csv("prompt_response_dataset.csv")
evaluation_df = pd.DataFrame(columns=["prompt", "response", "score", "reasoning"])
for _, row in df.iterrows():
    eval_result = evaluate_answer_relevance(prompt=row["prompt"], response=row["response"])
    print(eval_result)
    evaluation_df = evaluation_df.append({
        "prompt": row["prompt"],
        "response": row["response"],
        "score": eval_result.score,
        "reasoning": eval_result.reasoning
    }, ignore_index=True)
    time.sleep(20)


evaluation_df.to_csv("evaluated_responses.csv", index=False)
print("All done")

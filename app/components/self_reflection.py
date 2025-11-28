import re

def hallucination_self_reflection(llm, context, question, answer):
    """
    Uses the LLM to self-assess hallucination in its answer based on the provided context.
    Returns a score (0-1) and an explanation string.
    """
    reflection_prompt = f"""
You are a medical expert. Carefully review the following context and answer:

Context:
{context}

Question:
{question}

Answer:
{answer}

Instructions:
    Only answer 'No' if every fact in the answer is directly supported by the context above. If any part is not supported, answer 'Yes'.
    Respond in the format: Yes/No. Score: <number between 0 and 1>. Explanation: <your reasoning>.
Example response: No. Score: 0. Explanation: All facts are supported by the context.
"""
    reflection_result = llm.invoke(reflection_prompt)
    match = re.search(r"Score:\s*([0-1](?:\.\d+)?)\.?\s*Explanation:\s*(.*)", str(reflection_result))
    if match:
        score = float(match.group(1))
        explanation = match.group(2).strip()
    else:
        score = None
        explanation = str(reflection_result)
    return score, explanation

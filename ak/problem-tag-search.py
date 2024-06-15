import pickle
import numpy as np
from os.path import dirname, join
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import asyncio
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def get_llm(model: str):
    if model == "cleverchain-gpt-4o":
        openai_gpt4_args = {
            "seed": 123,
            # "function_call": None,
            # "response_format": {"type": "json_object"},
        }
        return ChatOpenAI(
            model="gpt-4o-2024-05-13",
            temperature=0.5,
            max_tokens=2000,
            model_kwargs=openai_gpt4_args,
            openai_api_key='sk-proj-Zwf0tSiDacdeiQRTMLVHT3BlbkFJMbreGiCSIpGX1rwYEtHD'
        )


# Asynchronous function to process each problem description
async def process_description(question, prompt):
    sys_msg = prompt
    system_message = SystemMessage(content=sys_msg)

    task = f"""
    ---

    This is the first competition problem, TASK 1:
    ```
    {question}
    ```

    """
    initial_human_message = HumanMessage(content=task)
    messages = [system_message, initial_human_message]
    llm = get_llm("cleverchain-gpt-4o")
    return await llm.ainvoke(messages)


class Problem_Matcher:
    def __init__(self, num_neighbors):
        current_dir = dirname(__file__)

        embeddings = np.load(join(current_dir, 'problem_embeddings.npy'))

        self.nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(embeddings)

        self.open_ai_client = OpenAI(api_key='sk-proj-Zwf0tSiDacdeiQRTMLVHT3BlbkFJMbreGiCSIpGX1rwYEtHD')

        with open(join(current_dir, 'simplified_df.pkl'), 'rb') as f:
            self.templates = pickle.load(f)

        with open(join(current_dir, 'simplify_task_prompt.txt'), 'r') as f:
            self.simp_prompt = f.read()

    async def get_solutions(self, problem_statement):
        simplified_problem = await process_description(problem_statement, self.simp_prompt)

        response = self.open_ai_client.embeddings.create(
            input=simplified_problem.content,
            model='text-embedding-3-small'
        )
        
        _, indices = self.nbrs.kneighbors([response.data[0].embedding])

        return self.templates['Solution'].iloc[indices[0]]


async def main():
    prob_match = Problem_Matcher(2)

    current_dir = dirname(__file__)

    with open(join(current_dir, 'test_case.txt'), 'r') as f:
        problem_statement = f.read()

    template_solutions = await prob_match.get_solutions(problem_statement)

    for row in template_solutions:
        print(highlight(row, PythonLexer(), Terminal256Formatter(style='monokai')))
        print("-*" * 40)
        print("-*" * 40)
        print("-*" * 40)


if __name__ == "__main__":
    asyncio.run(main())

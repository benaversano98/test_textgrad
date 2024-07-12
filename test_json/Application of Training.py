from loss import MultiFieldTokenParsedEvaluation
import textgrad as tg
import pandas as pd
import random
import json
import os

def load_dataset(filepath, evaluation_api, train_size=0.2, val_size=0.4, test_size=0.4):
    # Verificare che le dimensioni di train, val e test sommino a 1
    if train_size + val_size + test_size != 1.0:
        raise ValueError("Le dimensioni di train, val e test devono sommare a 1.0")

    # Caricare i dati JSON
    with open(filepath, "r") as f:
        data = json.load(f)

    # Convertire i dati in un DataFrame Pandas
    data = pd.DataFrame(data)

    # Calcolare gli indici di divisione per gli split
    total_data = len(data)
    train_end = int(total_data * train_size)
    val_end = train_end + int(total_data * val_size)

    # Dividere i dati in base agli indici
    train_data = tuple(data.iloc[:train_end].itertuples(index=False, name=None))
    val_data = tuple(data.iloc[train_end:val_end].itertuples(index=False, name=None))
    test_data = tuple(data.iloc[val_end:].itertuples(index=False, name=None))
    role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
    evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>."
    eval_instruction = tg.Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
    eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            role_descriptions=role_descriptions,
            engine=evaluation_api,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )

    return train_data, val_data, test_data, eval_fn

llm_engine = tg.get_engine("gpt-3.5-turbo")
tg.set_backward_engine("gpt-4o")

name_file = "word_sorting"
path = f"test_json/{name_file}.json"
_, val_set, _, eval_fn = load_dataset(path, llm_engine)
if os.path.exists(f"test_json/sys_json/system_prompt_{name_file}.json"):
        with open(f"test_json/sys_json/system_prompt_{name_file}.json", "r", encoding="UTF-8") as file:
            dati_json = json.load(file)
        dati_json_ordinati = sorted(dati_json, key=lambda x: x['val_performance'], reverse=True)
        system_prompt = dati_json[0]["system_prompt"]
else:
    system_prompt= "You are a concise LLM. Think step by step."


system_prompt = tg.Variable(system_prompt,
                            requires_grad=True,
                            role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))
# question_str, answer_str = val_set[0]
list_question = random.sample(val_set, 5)
for i, elem in enumerate(list_question):
    print(f"Stampa elemento n. {i+1}")
    print()
    question_str, answer_str = elem
    question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
    answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)
    print(question)
    print(answer)
    print()
    prediction = model(question)
    loss = eval_fn(inputs=[prediction, question, answer])
    if "0" in loss.value:
        while not "1" in loss.value:
            loss.backward()
            optimizer.step()
            prediction = model(question)
            loss = eval_fn(inputs=[prediction, question, answer])
    print(prediction)
    print()


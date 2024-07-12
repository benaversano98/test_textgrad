from loss import MultiFieldTokenParsedEvaluation
from textgrad.tasks import DataLoader
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
import numpy as np
import pandas as pd
import random
import json

# Carica le variabili d'ambiente da un file .env
load_dotenv(override=True)

# Funzione per impostare il seed per numpy e random, in modo da ottenere risultati riproducibili
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Funzione caricamento dati da json
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

# Funzione per valutare se una risposta a una domanda nel prompt è corretta
def eval_sample(item, eval_fn, model):
    # Estrai l'input (x) e l'output corretto (y) dal campione
    x, y = item
    
    # Crea una variabile tg.Variable per l'input con `requires_grad=False`
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    
    # Crea una variabile tg.Variable per l'output corretto con `requires_grad=False`
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    
    # Ottieni la risposta del modello all'input
    response = model(x)
    
    try:
        # Tenta di calcolare la valutazione utilizzando un dizionario di input
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        
        # Ritorna il valore della valutazione convertito in intero
        return int(eval_output_variable.value)
    except:
        # Se il primo tentativo fallisce, utilizza una lista di input
        eval_output_variable = eval_fn([x, y, response])
        
        # Fa parsing, cioè analizza e interpreta l'output della valutazione
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        
        # Ritorna il valore della valutazione analizzata convertito in intero
        return int(eval_output_parsed)

# Funzione per valutare un intero dataset di test
def eval_dataset(test_set, eval_fn, model, max_samples=None):
    # Se max_samples non è specificato, utilizza la lunghezza del test_set
    if max_samples is None:
        max_samples = len(test_set)
    
    # Lista per memorizzare le accuratezze
    accuracy_list = []
    
    # Crea un ThreadPoolExecutor (facilita l'esecuzione di operazioni parallele utilizzando un pool di thread) per eseguire valutazioni in parallelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []  # Lista per memorizzare i future objects delle valutazioni
        
        # Invia le valutazioni in parallelo
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            
            # Interrompe il ciclo se è stato raggiunto il numero massimo di campioni
            if len(futures) >= max_samples:
                break
        
        # Crea un tqdm loader per monitorare l'avanzamento delle valutazioni
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        
        # Per ogni future completato, ottieni il risultato e aggiorna l'accuratezza
        for future in tqdm_loader:
            acc_item = future.result()  # Ottieni il risultato del future
            accuracy_list.append(acc_item)  # Aggiungi l'accuratezza alla lista
            # Aggiorna la descrizione della barra di progresso con l'accuratezza media corrente
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    
    # Ritorna la lista delle accuratezze
    return accuracy_list 

# Funzione per eseguire la validazione e ripristinare il prompt del sistema se necessario
def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    # Valuta la performance del modello sul set di validazione
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    # Recupera la performance precedente dal dizionario dei risultati
    previous_performance = np.mean(results["validation_acc"][-1])
    
    # Stampa le performance corrente e precedente per il confronto
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    
    # Recupera il valore del prompt precedente dal dizionario dei risultati
    previous_prompt = results["prompt"][-1]
    
    # Se la performance attuale è inferiore a quella precedente, ripristina il prompt precedente
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")  # Stampa il prompt attuale che verrà rifiutato
        system_prompt.set_value(previous_prompt)  # Ripristina il valore del prompt precedente
        val_performance = previous_performance  # Imposta la performance attuale come quella precedente

    # Aggiungi la performance di validazione corrente (o ripristinata) al dizionario dei risultati
    results["validation_acc"].append(val_performance)
    return val_performance

# Imposta il seed per garantire la riproducibilità
set_seed(12)

# Ottieni l'engine per l'API di valutazione
# "gpt-4o" rappresenta un'API per il modello GPT-4 ottimizzato per le operazioni di valutazione
llm_api_eval = tg.get_engine(engine_name="gpt-4o")

# Ottieni l'engine per l'API di test
# "gpt-3.5-turbo-0125" rappresenta un'API per il modello GPT-3.5 turbo, utilizzato per le operazioni di test
llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo-0125")

# Imposta l'engine per il calcolo del gradiente inverso (backward) utilizzando l'API di valutazione
# "llm_api_eval" rappresenta il modello GPT-4 ottimizzato per la valutazione
# L'opzione override=True consente di sovrascrivere eventuali impostazioni precedenti dell'engine per il calcolo del gradiente inverso
tg.set_backward_engine(llm_api_eval, override=True)

# Definizione del file e del percorso del file
name_file = "object_counting"
path = f"test_json/{name_file}.json"
# Carica i dati e la funzione di valutazione per il task specifico
train_set, val_set, test_set, eval_fn = load_dataset(path, llm_api_eval)

print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))

print("Vuoi settare il system prompt da un file?")
try:
    while True:
        risposta = input("Si/No: ").capitalize().strip()
        if risposta != "Si" and risposta != "No":
            raise Exception("Risposta non valida")
        elif risposta == "Si":
            with open(f"test_json/sys_json/system_prompt_{name_file}.json", "r", encoding="UTF-8") as file:
                dati_json = json.load(file)
            dati_json_ordinati = sorted(dati_json, key=lambda x: x['val_performance'], reverse=True)
            STARTING_SYSTEM_PROMPT = dati_json[0]["system_prompt"]
            break
        else:
            STARTING_SYSTEM_PROMPT = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
            break
except Exception as e:
    print(e)
    
# STARTING_SYSTEM_PROMPT = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
# STARTING_SYSTEM_PROMPT = "You will need to reason about a problem posed by a user. Think step by step and explain how you would solve the problem. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

print(STARTING_SYSTEM_PROMPT)
train_loader = DataLoader(train_set, batch_size=3, shuffle=True)

# Creazione di una variabile tg.Variable per il prompt del sistema iniziale
# Questa variabile sarà utilizzata come prompt di sistema per il modello di valutazione
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,  # Indica che il gradiente sarà calcolato per questa variabile durante l'addestramento
                            role_description="system prompt to the language model")  # Descrizione del ruolo della variabile

# Creazione di un modello di valutazione utilizzando tg.BlackboxLLM con l'API di valutazione e il prompt del sistema
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

# Creazione di una nuova variabile tg.Variable per il prompt del sistema iniziale
# Questa variabile sarà utilizzata come prompt di sistema per il modello di test
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,  # Indica che il gradiente sarà calcolato per questa variabile durante l'addestramento
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")  # Descrizione del ruolo della variabile

# Creazione di un modello di test utilizzando tg.BlackboxLLM con l'API di test e il prompt del sistema
model = tg.BlackboxLLM(llm_api_test, system_prompt)

optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

# Inizializza i risultati con le performance iniziali
results = {"test_acc": [], "prompt": [], "validation_acc": []}
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())
print()

epochs = 3
# Esegui il ciclo di addestramento per x epoche
for epoch in range(epochs):  # Esegui per x epoche
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))): # tqdm per creare una barra di progresso
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")  # Aggiorna la descrizione della barra di progresso
        optimizer.zero_grad()  # Resetta i gradienti dell'ottimizzatore
        losses = []  # Inizializza una lista per accumulare le perdite
        
        for (x, y) in zip(batch_x, batch_y):  # Itera sui batch di dati di addestramento
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")  # Crea una variabile per l'input
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")  # Crea una variabile per l'output corretto
            response = model(x)  # Ottieni la risposta dal modello
            
            try:
                # Prova a calcolare la perdita utilizzando il dizionario di input
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                # Se fallisce, usa la lista di input
                eval_output_variable = eval_fn([x, y, response])
            
            losses.append(eval_output_variable)  # Aggiungi la perdita alla lista
        
        total_loss = tg.sum(losses)  # Somma tutte le perdite del batch
        total_loss.backward()  # Calcola il gradiente rispetto alla perdita totale
        optimizer.step()  # Aggiorna i parametri del modello

        # Esegui la validazione e ripristina il prompt del sistema se necessario e restituisce il val_performance
        val_performance = run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        
        print("sys prompt: ", system_prompt)  # Stampa il prompt di sistema corrente
        
        # Valuta le performance sul set di test
        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append(test_acc)  # Aggiungi l'accuratezza del test ai risultati
        results["prompt"].append(system_prompt.get_value())  # Salva il valore corrente del prompt
        
        if steps == 3:  # Se sono stati completati 3 passi, esci dal ciclo
            break

# Salvataggio del prompt in un file JSON
json_file = {"system_prompt" : str(system_prompt.get_value()), "epoche" : epochs, "val_performance" : val_performance}

try:
    with open(f"test_json/sys_json/system_prompt_{name_file}.json", "r", encoding="UTF-8") as file:
        json_prompt = json.load(file)
except FileNotFoundError:
    json_prompt = []

json_prompt.append(json_file)

with open(f"test_json/sys_json/system_prompt_{name_file}.json", "w", encoding="UTF-8") as outfile:
    json.dump(json_prompt, outfile, indent=4)

# EXTRA
# Controllo del system_prompt ottimizzato

# question_str, answer_str = val_set[0]
# question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
# answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)

# print(question)
# print(answer)

# # model = tg.BlackboxLLM(llm_api_test, system_prompt=system_prompt)

# prediction = model(question)

# print()
# print(question)
# print(prediction)


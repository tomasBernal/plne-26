import torch
import json
import pandas
import random

import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from evaluate import load
from sklearn.metrics import classification_report

# Modelo BERT en español - BETO
#path_bert_model = 'dccuchile/bert-base-spanish-wwm-uncased'

# Modelo BERT multilingüe
#path_bert_model = 'bert-base-multilingual-cased'

# Modelo BERTIN basado en RoBERTa
#path_bert_model = 'bertin-project/bertin-roberta-base-spanish'

# Modelo MrBERT basado en BERT
path_bert_model = 'BSC-LT/MrBERT-es'

# Modelo "destilados" de BERT en español
#path_bert_model = 'CenIA/distillbert-base-spanish-uncased'

# Modelo AlBERT en español
#path_bert_model = 'CenIA/albert-base-spanish'

SEED = 42

# Cargamos un modelo de BERT preentrenado para clasificación. El número de etiquetas es 2
NUM_LABELS = 2

# Definimos la configuración de las labels e ids
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# La clase TFAutoModelForSequenceClassification es de Tensorflow y la AutoModelForSequenceClassification es de Pytorch
bert_class_model = AutoModelForSequenceClassification.from_pretrained(path_bert_model, num_labels=NUM_LABELS)
bert_class_model.config.id2label = id2label
bert_class_model.config.label2id = label2id

# Cargamos el Tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_bert_model)


# Cargamos los conjuntos de entrenamiento y test
df_train = pandas.read_csv("dataset_train.csv",encoding="UTF-8")
df_test = pandas.read_csv("dataset_test.csv",encoding="UTF-8")

p_train = 0.80 # Porcentaje de train.
p_eval = 0.20 # Porcentaje de eval.

# Mezclamos el dataset de manera aleatoria para poder obtener luego el conjunto
# de desarrollo "dev"
df_train.sample(frac=1, random_state=SEED)

# Para poder entrenar es necesario codificar las etiquetas como números. Para eso codificaremos
# los negativos con 0 y los positivos con 1 según lo definido en los diccionarios
# label2id e id2label
df_train['_label'] = df_train['label'].apply(lambda x: label2id[x])
df_test['_label'] = df_test['label'].apply(lambda x : label2id[x])

df_train, df_eval = train_test_split (df_train, test_size = p_eval)

print("Ejemplos usados para entrenar: ", len(df_train))
print("Ejemplos usados para evaluar: ", len(df_eval))
print("Ejemplos usados para test: ", len(df_test))


# Fijar semilla para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Cargamos el modelo para clasificación en Pytorch a través de una función para asegurar reproducibilidad
bert_class_model_pytorch = AutoModelForSequenceClassification.from_pretrained(path_bert_model, num_labels=NUM_LABELS)
bert_class_model_pytorch.config.id2label = id2label
bert_class_model_pytorch.config.label2id = label2id

# Los datasets se preparan de manera distinta a Tensorflow
class PLNEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(classification_report(labels, predictions, digits=6)) # Ver informe de clasificación
    return metric.compute(predictions=predictions, references=labels)

batch_train_size = 16
batch_val_size = 64
metric_name = "eval_f1"

# Definimos algunos training arguments como el tamaño del bach_size
training_args = TrainingArguments (
  output_dir = './results',
  logging_dir = './logs',
  num_train_epochs=3,  # Número de épocas
  eval_strategy="epoch",  # Estrategia de evaluación (epoch / step)
  save_strategy="epoch",
  per_device_train_batch_size = batch_train_size,
  per_device_eval_batch_size = batch_val_size,
  metric_for_best_model=metric_name,  # Seleccina el mejor modelo según esta métrica
  save_total_limit=1, # Almacena el número de checkpoints
  load_best_model_at_end=True, # Carga el mejor modelo al final
  report_to="none"  # Desactiva wandb
)

tokenized_train_dataset = tokenizer (df_train.tweet.tolist (),  truncation=True, padding = True)
tokenized_eval_dataset = tokenizer (df_eval.tweet.tolist (), truncation=True, padding = True)
tokenized_test_dataset = tokenizer (df_test.tweet.tolist (), truncation=True, padding = True)


# Como antes, las etiquetas deben ser numéricas para poder entrenar.
# Preparamos los 3 datasets para hacer el finetuning
train_dataset = PLNEDataset (tokenized_train_dataset, df_train._label.tolist())
eval_dataset = PLNEDataset (tokenized_eval_dataset, df_eval._label.tolist())
test_dataset = PLNEDataset (tokenized_test_dataset, df_test._label.tolist())

# Usar DataCollator para padding dinámico
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer (
    model = bert_class_model_pytorch,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
    data_collator=data_collator
)
trainer.train()


print ("PREDICCIONES SOBRE EVAL")
bert_class_model_pytorch.eval()
print (json.dumps (trainer.evaluate (), indent = 2))


# Salvamos el modelo reentrenado
modelo ='modeloFinetuningPytorch'
trainer.save_model (modelo)
tokenizer.save_pretrained (modelo)

print ("PREDICCIONES SOBRE TEST")
predictions = trainer.predict (test_dataset)
print(json.dumps(predictions.metrics, indent = 2))
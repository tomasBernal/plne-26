import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import json
from transformers import AutoModelForSequenceClassification


# Salvamos el modelo reentrenado
modelo ='modeloFinetuningPytorch'
NUM_LABELS = 2

# Definimos la configuración de las labels e ids
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# Cargamos el modelo ya con el FineTuning hecho
bert_class_model = AutoModelForSequenceClassification.from_pretrained(modelo, num_labels=NUM_LABELS)
bert_class_model.config.id2label = id2label
bert_class_model.config.label2id = label2id
# Cargamos el Tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelo)

# Probamos a clasificar estas frases
textos = ['hay muchos más muertos por covid',
          'el número de afectados por covid aumenta',
          'vamos a salir de la pandemia',
          'ánimo a todos',          
]


# TEST

for text in textos:
  inputs = tokenizer(text, return_tensors="pt")
  with torch.no_grad():
    logits = bert_class_model(**inputs).logits
  predicted_class_id = logits.argmax().item()
  prediction= bert_class_model.config.id2label[predicted_class_id]
  print(text,'=>', predicted_class_id, '=>', prediction, "  ", logits.softmax(1))
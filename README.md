# PLNE 2026
# Procesamiento de Lenguaje Natural Escrito

Este repositorio contiene material docente y práctico para la asignatura **Procesamiento de Lenguaje Natural Escrito (PLNE)** del curso **2026–27**.

## ATLASv2
Esta carpeta contiene recursos de **apoyo docente y técnico** para aprender a trabajar en el servidor **ATLASv2**.

### ¿Qué es ATLASv2?
ATLASv2 es un **clúster** que permite ejecutar:
- Entrenamiento de modelos de Machine Learning y Deep Learning.
- Experimentos con GPU.
- Procesos costosos en tiempo o memoria.

Que no serían viables en un ordenador personal.

En lugar de ejecutar programas directamente, los usuarios **envían trabajos (jobs)** al sistema, indicando:
- Qué recursos necesitan (GPU, memoria, tiempo).
- Qué programa quieren ejecutar.

### ¿Qué es Slurm y por qué se usa?
**Slurm** es un **gestor de colas y recursos** que se encarga de:
- Asignar CPUs, GPUs y memoria.
- Planificar la ejecución de los trabajos.
- Evitar que varios usuarios interfieran entre sí.

### Comandos slurm que debeís conocer
- sbatch: para lanzar trabajos, indicando el script a ejecutar.
- squeue: para consultar la cola de trabajos.
- scancel: para cancelar un trabajo, indicando su PID.

### Idea general de los scripts en ATLASv2
Esta carpeta tiene recursos de **material de apoyo docente y técnico** para aprender a:
- ejecutar trabajos con GPU usando Slurm,
- trabajar con las carpetas temporales `/scratch`, y con el HOME.
- entrenar y usar modelos de Hugging Face para clasificación de textos
- aplicar distintos paradigmas de aprendizaje de In-Context Learning (zero-shot, few-shot, chain-of-thought).

## Estructura general del repositorio
La estructura principal del repositorio es la siguiente:

```text
plne-26/
├── README.md
└── atlasv2/
    ├── bootstrap.sh
    ├── P8/
    │   ├── dataset_train.csv
    │   ├── dataset_test.csv
    │   ├── finetuning.py
    │   └── inference.py
    └── P9/
        ├── atlas_utils.py
        ├── prompting_utils.py
        ├── zero_shot.py
        ├── few_shot.py
        └── chain_of_thought.py

```

Descripción por carpetas y ficheros:

- `README.md`  
  Documento principal del repositorio.

- `atlasv2/bootstrap.sh`  
  Script de lanzamiento para enviar trabajos al clúster ATLASv2 usando Slurm.
    ```
    Ejemplo: sbatch bootstrap.sh zero_shot.py
    ```

- `atlasv2/P8/` (Práctica 8)  
  Contiene datos y scripts para fine-tuning e inferencia de un modelo encoder-only:
  - `dataset_train.csv`: conjunto de entrenamiento.
  - `dataset_test.csv`: conjunto de test.
  - `finetuning.py`: fine-tuning de modelos encoder-only.
  - `inference.py`: generación de predicciones/inferencia.

- `atlasv2/P9/` (Práctica 9)  
  Scripts de utilidades y prompting para LLMs:
  - `atlas_utils.py`: utilidades comunes para entorno ATLASv2.
  - `prompting_utils.py`: funciones auxiliares para construir prompts y procesar salidas.
  - `zero_shot.py`: script para experimentos de prompting tipo zero-shot.
  - `few_shot.py`: script para experimentos de prompting tipo few-shot.
  - `chain_of_thought.py`: script para experimentos de prompting tipo chain-of-thought.


## Uso de Hugging Face y `HF_TOKEN`

Algunos modelos (**Llama** y **Gemma**, entre otros) están protegidos (*gated models*) y requieren autenticación.

### Cómo obtener un token de Hugging Face
1. Crear una cuenta o iniciar sesión en Hugging Face:
   - https://huggingface.co
2. Ir a:
   - https://huggingface.co/settings/tokens
3. Crear un token (con permisos de lectura es suficiente).

Pon el token en el fichero ```bootstrap.sh``` en la variable ```HF_TOKEN```

```
  export HF_TOKEN="tu_token"
```
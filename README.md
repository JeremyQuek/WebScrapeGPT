# RAG EVALUATION PIPELINE ⚙️    

![Python](https://img.shields.io/badge/Python-555555?style=for-the-badge&logo=python&logoColor=white)
![3.12](https://img.shields.io/badge/3.12-FFC11A?style=for-the-badge)


## Introduction
The RAG (Retrieval-Augmented Generation) Evaluation Pipeline is designed to evaluate the performance of machine learning 
models by leveraging LLMs to provide detailed metric-based analysis. This pipeline aims to ensure that our models meet 
the required standards for deployment and can effectively perform their intended tasks in real-world scenarios.

<div align="center">
  <kbd>
    <img src="3.png" alt="drawing" width="600"/>
  </kbd>
</div>


## Table of Contents

- [Installation](#installation)
- [Introduction](#introduction)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

## Installation

No installation is required for this project. Simply clone the repository and open the `index.html` file in your web browser.


```
git clone https://github.com/yourusername/center-items-flexbox.git
cd center-items-flexbox
```


## Usage

This pipeline provides two different classes for use: `GenerateTestset` and `EvaluateModel`.


## Evaluate Model

As the name suggest, the `EvaluateModel` calculates various metrics based on the user provided testset.

This class requires questions, answers, retrieved contexts, and ground truths compiled.
If you would like to generate a test case for your model, see`GenerateTestset`. 

The answers, questions, retrieved context and ground truths must either be a python `dictionary` or `pandas.Dataframe`.


Format:

```python
results = {'question':       [ ],
            'answer':        [ ],
            'contexts':     [[ ]] ,
            'ground_truth':  [ ]
            
}
```
Example:
```
question: ['When was Einstien born?', 'Where is Singapore']

contexts: [ ['Miracle year..', 'Germany..'], ['Singapore located in..', 'Asia..'] ]

ground_truth: ['1879', 'South-East Asia']

answer: ['1879', 'South Asia']
```

Note: Pay close attention to the spelling of the keys (`contexts` not `context`)

Also note that `contexts` is **nested list**.


### Other supported file types

<details>
<summary>Click to show more</summary>

If you have compiled your testset in one file and want to transfer it elsewhere, 
you may use the following methods to transfer files.

Pickle file:
```python
import pickle

"results.pkl"
report = EvaluateModel(dataset = "results.pkl")
```
CSV:
```python
import csv

"results.csv"
report = EvaluateModel(dataset = "results.csv")


##All pickle files and csv files must still contain a dictionary with appropriately named columns.
```

Note: No additional steps are required to preserve the integrity of `lists` when saving as a csv file. 
You can directly save your results into a csv and`EvaluateModel` will take care of parsing the `lists` from strings in your data.

</details>

<br>

### Generating Report
Once you have passed your dataset into `EvaluateModel`, you may do `.get()` to generate a report.
```python
results = EvaluateModel(dataset = 'results.csv')
report = results.get()
```
This will generate both a .html report card and .xlsx dataset which you can save

<details>
<summary>Click to show example HTML and XLSX</summary>
<br>

HTML Report Card:


![EXCEL FILE](/Users/jeremyquek/PycharmProjects/pythonProject/Final Eval Package/Images/Screenshot 2024-07-01 at 11.02.25 AM.png)

<br>

EXCEL sheet:
![EXCEL FILE](/Users/jeremyquek/PycharmProjects/pythonProject/Final Eval Package/Images/Screenshot 2024-07-01 at 11.20.45 AM.png)
</details>

Additionally you may also do `print()` to get the results in your terminal window
```python
print(report)
```
```python
#Print Result:

Calculating scores...
Metrics: {'answer_correctness': 0.7018, 'faithfulness': 0.8841, 'answer_relevancy': 0.9446, 'context_precision': 0.9792, 'context_recall': 0.9437, 'BERT': 0.9097, 'Rouge': 0.4151, 'MRR': 0.5000}
```




### Additional Configurable Options:





### 1) Metrics

<details>
<summary>Click to show more</summary>

By default this provide provides a total of 8 different metrics for assessing various components of the RAG pipeline. 5 of which are
[RAGAS metrics](https://docs.ragas.io/en/latest/concepts/metrics/index.htmle), which is an open-source RAG evaluation framework. 
These metrics assess the various components of the pipeline as shown below:

**RAGAS Metrics**

1) Answer Correctness: Assesses the factual accuracy of your model's answer to the ground truth


2) Answer Relevancy: Assesses how well your model's answer addresses the initial question


3) Faithfulness: Assesses how faithful your model's answer is to the reference context


4) Context Recall: Assesses how relevant each retrieved item is to the ground truth


5) Context Precision: Assesses how many relevant items in total are present in the retrieved contexts 


By default, if no `ragas_metrics` parameter is passed, all metrics are will be computed and displayed.
You may configure this to not display certain metrics if you wish.

Example:
```python
#1) Default parameter settings:

ragas_metrics = [
        answer_correctness,
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
  
#2) Example custom parameter settings:

from ragas.metrics import (
    context_recall,
    context_precision,
    answer_correctness
)
  

results = EvaluateModel(dataset = 'results.csv', ragas_metrics = [answer_correctness])
```

**OTHER Metrics**

To provide  a holistic assessment of a RAG model, this pipeline includes 3 other metrics from various sources
that assess different components of a RAG pipeline. These are as shown:

1) BERTScore: Bert score is a metric calculated using BERT embeddings to assess the quality of generated answer to the reference ground truth. 
This score is calculated using Facebook's advanced. [(Read more)](https://huggingface.co/FacebookAI/roberta-large)


2) Rouge: Rouge score is a metric that assesses the quality of generated answer against a reference context based on the longest common sequence (LCS) between the two texts. [(Read more)](https://github.com/google-research/google-research/tree/master/rouge).


3) Mean Reciprocal Ranking (MRR): The MRR score reflects the ability of the retriever to prioritise 
relevant documents in the top ranked results. It is a measure of the reciprocal of the rank of the first relevant document. [(Read more}](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr)


By default, if no other_metrics parameter is passed, all metrics are will be computed and displayed. You may configure this to not display certain metrics if you wish.

Example
```python
#1) Default parameter settings:

other_metrics = [
        "BERT",
        "ROGUE",
        "MRR"
    ]
  
  
#2) Example custom parameter settings:

results = EvaluateModel(dataset = 'results.csv', other_metrics = ["MRR"])
```
</details>







### 2) Critical LLM
<details>
<summary>Click to show more</summary>


The critical LLM is the language model used for the assessment of the various metrics of your report. 6 out of the 8 metrics
are assessed with the use of a model. These are RAGAS metrics and the BERTScore.

**BERT Model**

Bert score is calculated using facebook's RoBERTa-large model to generate embeddings and compute semantic similarity.
The RoBERTa model is embedded in AWS EC2 as part of the overall framework, and is not configurable.

**RAGAS Default Model**

The 5 ragas metrics are computed using the default model choice in built with the RAGAS evaluation package. This is open-AI's 
gpt-3.5-turbo large language model. However, this option can be configured by the user if they wish.
Instructions to load and bring your model can be found on RAGAS's website.

As this pipeline is hosted on AWS, there are several ways to utilise a custom model as shown below:

<br>

### Bring your own Models


**2.1) With LangChain**

Langchain offers a way to load in custom models for use with RAGAS evaluations. For more information please visit the 
[RAGAS website](https://docs.ragas.io/en/latest/howtos/customisations/bring-your-own-llm-or-embs.html).


Simple implemention of custom model using LangChain .
```
pip install langchain 
```

```python
import os
from langchain_openai import ChatOpenAI

#Initialise Model
os.environ["OPENAI_API_KEY"] = 'sk-proj-...'

langchain_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


#Run Report
results = EvaluateModel(llm = langchain_llm, dataset = dataset)

report = results.get()
```
You can also customise the use of embeddings models
```python
import os
from langchain_openai import OpenAIEmbeddings

#Initialise Model
os.environ["OPENAI_API_KEY"] = 'sk-proj-...'

embeddings = OpenAIEmbeddings(model="text-embeddings-3-small")


#Run Report
results = EvaluateModel(embeddings = embeddings, dataset = dataset)

report = results.get()
```
<br>

**2.2) With Langchain Hugging Face**

Hugging face offers an alternate solution to load in custom open-source models with their transformers pipeline. 
Below is simple code implementation of utilisng a custom hugging face model.

```
pip install langchain-huggingface
```

Importing hugging face models and embeddings
```python
from langchain_huggingface import ChatHuggingFace

from langchain_huggingface import HuggingFaceEmbeddings
```

Implement as before.
```python
#Insert your access authentication token code for hugging face**


#Initialise Models

embeddings = HuggingFaceEmbeddings(model= "BAAI/bge-small-en-v1.5")

llm = ChatHuggingFace(model="Mistral-7B-v0.1")


#Run Report
results = EvaluateModel(llm = llm, embeddings = embeddings, dataset = dataset)

report = results.get()
```
<br>

**2.3) With RAGAS Critic Model, AzureOpenAI, GCP Vertex or AWS Bedrock**

For more information on integrations with more different models please visit the [RAGAS website](https://docs.ragas.io/en/latest/howtos/customisations/bring-your-own-llm-or-embs.html).
</details>


## Example

Full code implemention:

```python
import EvaluateModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import (faithfulness, answer_correctness)
import os

#Initliase models
os.environ["OPENAI_API_KEY"] = 'sk-proj-'
langchain_llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="davinci-002")


#Define the variables
dataset = "test_results.pkl"
ragas_metrics = [faithfulness, answer_correctness]
other_metrics = ["BERT", "MRR"]

#Run report
results = EvaluateModel(
    llm = langchain_llm,
    embeddings = embeddings,
    dataset = dataset,
    ragas_metrics = ragas_metrics,
    other_metrics = other_metrics
)

report = results.get()
```

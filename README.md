## AIMO Solver

This repository contains code corresponding to the paper:

**“Improving Math Problem Solving in Large Language Models Through Categorization and Strategy Tailoring”**  
– appeared at **AICCONF 2025**, and available on arXiv (arXiv:2411.00042) :contentReference[oaicite:0]{index=0}.

### BibTeX
```bibtex
@article{akella2024improving,
  title={Improving Math Problem Solving in Large Language Models Through Categorization and Strategy Tailoring},
  author={Akella, Amogh},
  journal={arXiv preprint arXiv:2411.00042},
  year={2024}
}

---

## Overview

This repository implements the methods proposed in the paper: categorizing AIME-level math problems into four classes (Algebra, Combinatorics, Geometry, Number Theory), then applying category-specific prompting strategies—Chain-of-Thought (CoT) or Program-of-Thought (PoT)—to improve the solver’s accuracy and reliability.

The pipeline comprises three core files:

* **`subj_data.py`**
  Contains your *custom-curated* AIME-level problem datasets. Defines four functions:

  * `get_geo()` – returns geometry problems
  * `get_alg()` – returns algebra problems
  * `get_nt()` – number theory problems
  * `get_cp()` – combinatorics problems

* **`subject_categorization.py`**
  Implements a lightweight neural model for categorizing problems by subject.

  * `train_subj()` trains the model using your curated problem sets.
  * `test_subj(txt)` predicts the category label (`A`, `C`, `G`, `N`) for a new problem.

* **`solve_single_problem.py`**
  Defines the `AIMO_Solver` class, which orchestrates:

  1. **Subject classification** via `test_subj()`.

  2. **LLM prompting**—using either a Chain-of-Thought or Program-of-Thought approach depending on the category.

  3. **Multiple-run aggregation**: generating several answers per problem and selecting the most consistent one.

  > **Acknowledgment**: This code was **derived from** the Kaggle notebook [*Improved Code Interpretation*](https://www.kaggle.com/code/abdurrafae/improved-code-interpretation), adapted here to integrate problem categorization and curated datasets.

---

## Installation & Setup

### Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### Create `requirements.txt`

```text
torch
transformers
tqdm
numpy
tensorflow
sympy
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage Example

### Step 1: Train the subject classifier

```python
from subject_categorization import train_subj
train_subj()
```

### Step 2: Use the solver on a sample problem

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from solve_single_problem import AIMO_Solver

# Load a suitable LLM
model_name = "gpt2"  # Replace with an LLM capable of thoughtful reasoning
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

solver = AIMO_Solver(model, tokenizer, pipe)

problem = "How many positive integers less than 1000 are divisible by both 6 and 10?"
answer = solver.predict(problem)

print("Predicted Answer:", answer)
```

---

## Repository Structure

```
.
├── subj_data.py                 # Curated AIME-level problem datasets
├── subject_categorization.py   # Trains/tests a subject-classification model
├── solve_single_problem.py     # LLM-based solver with categorization & strategy
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

---

## Key Highlights

* **Custom-curated datasets** (`subj_data.py`) designed by you to improve subject-classification reliability.
* **Category-specific prompting strategies**:

  * *Chain-of-Thought (CoT)* for reasoning-heavy categories (e.g., geometry).
  * *Program-of-Thought (PoT)* for computation-intensive categories, via Sympy code generation.
* **Answer aggregation** across multiple runs to boost consistency and reduce hallucination.
* **Pipeline inherits from previous work**—solid foundation plus your unique enhancements.

---

## Acknowledgments

* **Paper**: *Improving Math Problem Solving in Large Language Models Through Categorization and Strategy Tailoring* – AICCONF 2025 ([arXiv][1]).
* **Code derivation**: based originally on the Kaggle notebook \[*Improved Code Interpretation*] by Abdur Rafae.
* **Your contribution**: Custom-curated datasets, subject-classifier, integration into LLM pipeline with tailored strategies.

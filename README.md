# Commonsense Validation and Explanation

SemEval 2020 Task 4: Commonsense Validation and Explanation. This repo is only for task a.

This task is inspired and extension of the ACL 2019 paper: Does It Make Sense? And Why? A Pilot Study for Sense Making and Explanation. [link](https://arxiv.org/abs/1906.00363)

Codalab competition [here](https://competitions.codalab.org/competitions/21080)!

## Introduction

The task is to directly test whether a system can differentiate natural language statements that make sense from those that do not make sense. The first task is to choose from two natural language statements with similar wordings which one makes sense and which one does not make sense

The detailed description of the task can be found in [Task Proposal](./TaskProposal.pdf).

### Example

#### Task A: Validation

Which statement of the two is against common sense?

- Statement 1: He put a turkey into the fridge. *(correct)*
- Statement 2: He put an elephant into the fridge.

## requirement:

pytorch 1.3

transformers 2.1.1

## run

```shell
python train.py
```
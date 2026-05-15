# Vetorização de Dados Discretos via Algoritmo de Schneider (1990)

Este projeto integra o **Tema 4: Recuperação de Modelos a partir de Dados**, com foco na vetorização de curvas de Bézier e B-Splines via aproximação. O objetivo central é recuperar curvas paramétricas a partir de dados discretos, como contornos e trajetórias, utilizando técnicas de ajuste fundamentadas em otimização numérica e álgebra linear, áreas fundamentais tanto para a Computação Gráfica quanto para o Aprendizado de Máquina.

## Descrição do Projeto

O projeto visa implementar e modernizar algoritmos de vetorização que utilizam interpolação e aproximação para a representação de curvas. A referência central é o método de **Philip J. Schneider (1990)**, que combina heurísticas de segmentação e mínimos quadrados para converter dados discretos em representações polinomiais por partes.

A base de estudo deste trabalho é a implementação de referência presente no repositório `fitCurves`, desenvolvida por Volker Poplawski em Python 2. O escopo do projeto compreende a refatoração e modernização deste código, visando superar limitações da época de estrutura e performance.

## Objetivos de Desenvolvimento

* **Modernização Tecnológica:** Migração e refatoração do código original de Python 2 para Python 3.10+.
* **Otimização Computacional:** Substituição de estruturas iterativas manuais por operações matriciais vetorizadas utilizando a biblioteca **NumPy**.
* **Refinamento Arquitetural:** Organização modular das funções de cálculo matemático, lógica de ajuste (*fitting*) e rotinas de visualização.

## Ferramentas

* **Linguagem:** Python 3
* **Processamento Numérico:** NumPy e SciPy
* **Visualização Técnica:** Matplotlib

## Referências

* **Schneider, P. J. (1990).** *An Algorithm for Automatically Fitting Digitized Curves*. In A. Glassner (Ed.), Graphics Gems. Academic Press.
* **volkerp (GitHub).** *fitCurves*: Python implementation of Schneider’s fitting algorithm. Disponível em: [https://github.com/volkerp/fitCurves](https://github.com/volkerp/fitCurves)

---
*Trabalho Prático desenvolvido para a disciplina de Computação Gráfica (2026.1) – Profa. Asla Medeiros de Sá.*

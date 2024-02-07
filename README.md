<div align='center'>
    <h1>French News Bot</h1>
    <h2>Acceder aux actualités de la  France dans plusieurs domaines.</h2>
    <em>Par <a href='https://www.linkedin.com/in/goudja-mahamat'>Goudja Mahamat </a></em>

</div>

## Table des matières
- [1. Introduction](#1-Introduction)
- [2. Modules du projet](#2-Modules-du-projet)
    - [2.1 Q&A data generator](#21-q-and-a-data-generator)
    - [2.2 Training pipeline](#22-Training-pipeline)
    - [2.3 Feature pipeline](#23-Feature-pipeline)
    - [2.4 News bot](#24-News-bot)
- [3. Configuration des services externes](#3-Configuration-des-services-externes)
    - [3.1. Alpaca](#31-alpaca)
    - [3.2. Qdrant](#32-qdrant)
    - [3.3. Comet ML](#33-comet-ml)
    - [3.4. Beam](#34-beam)
- [4. Installation et utilisation](#4-installation-et-utilisation)

-------------------------


## 1. Introduction

Ce projet est essentiellement basé sur [le travail](https://github.com/iusztinpaul/hands-on-llms) réalisé par <i> <a href="https://github.com/iusztinpaul">Paul Iusztin</a>, <a href="https://github.com/Paulescu">Pau Labarta Bajo</a> et <a href="https://github.com/Joywalker">Alexandru Razvant</a></i>. 
Ce ptojet correspond à la création d'un agent conversationnel avec lequel il est possible d'interagir en lui posant des questions sur les actualités en France. Il s'agir concretement de choisir une rubrique d'actualité (politics par exemple) dans liste proposée et poser une question pour laquelle le bot générera une réponse conhérente.

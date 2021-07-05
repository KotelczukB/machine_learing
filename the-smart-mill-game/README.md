# The Smart Mill Game

## Gliederung

- [Projektbeschreibung](#description)
- [Features](#features)
- [Technologien](#technologies)
- [How To Start](#how_to_start)
- [Präsentation](#presentation)
- [Video](#video)
- [Projektmitglieder](#members)


<a name="description"></a>

## Projektbeschreibung

Es soll ein Mühle-Feld inklusive des aktuellen Spielstandes erkannt werden. Anschließend soll die KI regelkonform spielen und sowohl versuchen, gegnerische Mühlen zu verhindern als auch eigene Mühlen zu generieren. 


<a name="features"></a>

## Features

* Erkennung des Spiels
	- Erkennung des Spielfeldes
	- Erkennung der aktuellen Konstellation (gegnerischer und eigener Spieler)
	- Erkennung der Spielphase (Legephase, normale Spielphase, Springphase)
* Spielen
  - Regelkonformes Spielen
  - Verhindern gegnerischer Mühlen und Legen eigener Mühlen durch Deep Reinforcement Learning


<a name="technologies"></a>

## Technologien

+ Tensorflow (2.3.0)
+ KerasRL
+ Stable_Baselines3
+ OpenCV
+ Tensorboard
+ Pillow
+ mathplotlib
+ graphviz
+ pydot
+ gym

### Camera Input und Mill Evaluation

+ Erstellen eines quadratischen Bildes des Mühle-Feldes (OpenCV)
+ Splitten des Bildes in 49 quadratische Segmente (Pillow, mathplotlib)
+ Erstellen des Observation Space mithilfe eines Keras-Models

### Custom Enviroment

+ Nach [OpenAI Gym](https://gym.openai.com/) Standard

###  Mill Agent
Auswertung des aktuellen Spielstandes und die Vorhersage der nächsten Aktion. 
+ Enviroment: Custom Enviroment nach OpenAI standard (OpenAI Gym)
+ Model: Sequential Keras Model
+ Agent:
	+ Keras DQN Agent
	+ Stable_Baselines3 PPO Agent
* Tensorflow 
* Tensorboard


<a name="how_to_start"></a>

## How To Start

Das [tsmg_presentation_1.2.ipynb](./mill_presentation/tsmg_presentation_1.2.ipynb) bildet eine lauffähige Version mit allen Funktionalitäten ab.


<a name="presentation"></a>

## Präsentation

[Präsentation vom 05.07.2021](./KS_Praesentation_2020-07-05.pdf)


<a name="video"></a>

## Video

... in Kürze


<a name="members"></a>

## Projektmitglieder

* Bartosz Kotelczuk
* Andreas Klinger

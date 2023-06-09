# CartPole DQN reinforcement learning
![](https://img.shields.io/badge/TensorFlow-v.%202.10.1-red) ![](https://img.shields.io/badge/Gym-v.%200.26.2-blue) ![](https://img.shields.io/badge/NumPy-v.%201.24.2-orange) ![](https://img.shields.io/badge/Matplotlib-v.%203.6.2-lightgrey)

Program sa používa na testovanie navrhnutého agenta pre Cartpole-v1 z knižnice Gym. 
Potrebné knižnice na spusteni:<br />
- TensorFlow v. 2.10.1  pip install tensorflow==2.10.1 
- Gym v. 0.26.2         pip install gym==0.26.2
- NumPy v. 1.24.2       pip install numpy=1.24.2
- Matlotlib v. 3.6.2    pip install matplotlib==3.6.2 
 
Funkcia trénovania je prispôsobená na fungovanie v Google Colaboratory a nemusí byť lokálne skompilovaná.
Pri spušťaní na počítači, odporúčam používať Conda prostredie, aby bolo možné použiť grafickú kartu (ak je nainštalovaná).

Stiahneme si repozitár s projektom.<br />

Postup inštalácie prostredia Conda nájdeme na webstránke: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html. <br />
Ak používame prostredie Conda, TensorFlow a ostatné knižnice inštalujeme priamo do prostredia podľa návodu na stránke: https://www.tensorflow.org/install/pip#windows-native_1 <br />
Ostané knižnice inštalujeme podľa hore uvedených príkazov.

Pre bežné systémy používame správcu rozšírení jazyka Python- pip.<br />
Otvorime si priečinok s projektom v príkazovom riadku. Môžme si priečinok otvoriť v prieskumníkovi a napísať do horného riadku, kde sa zobrazuje cesta k súboru príkaz "cmd", ktorý nám priamo otvorí príkazový riadok na aktuálnej ceste. Do tohto príkazového riadku, budeme postupne písať hore uvedené príkazy na inštaláciu. 

Otvoríme si projekt v nami zvolenom vývojárskom prostredí(IDE).<br />

Pri spúšťaní programu si môžme vybrať medzi dvoma nastaveniami:
- Trénovanie a následné testovanie: spustí sa cyklus trénovania a vyexportovanú, natrénovanú, sieť následne spustí a počas jednej epochy sa vytvorí záznam. Tento záznam sa uloží vo formáte GIF. Hodnota: False
- Testovanie: je nutné aby v koreňovom priečinku projektu sa nachádzal súbor s názvom "trained_model.h5", ktorý obsahuje váhy neurónovej siete. Program sa následne spustí a vytvorí GIF. Hodnota: True

Pre voľbu režimu je nutné prepísať na riadku 179, hodnotu premennej skip_training podľa hodnoty uvedenej pri type spustenia.

Pre zobrazenie generovaných grafov počas trénovania sa používa príkaz: "tensorboard --logdir logs/dqn"

Odkaz na GitHub repozitár: https://github.com/MatthewSVK/Reinforcement-learning-app

Odkaz na Google Colaboratory s projektom (je potrebný STU mail): https://colab.research.google.com/drive/1LZW6WZ8zw_Sl5GacOOI0G5szjNfdkvsY?usp=sharing

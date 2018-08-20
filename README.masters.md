Cilj naloge je izboljšati predhodno raziskano
napovedovanje porabe električne energije [1] z
uporabo sledečih globokih nevronskih mrež:
– konvolucijska nevronska mreža (+ neopredeljena
polno-konvolucijska nevronska mreža [2])
– rekurenčna nevronska mreža (+ LSTM - nevronska
mreža z dolgo-kratkoročnim spominom [6])
– ansambelski operatorji povprečja, mediane in
frekvence [4] nad omenjenimi nevronskimi
mrežami

Opravljeno delo in glavni rezultati
• Implementacija metode za preprocesiranje časovnih
vrst (eliminacija meritev > μ + 3σ, izgradnja
vrednostnih, časovnih in meteoroloških značilk,
normalizacija)
• Implementacija metode za progresivno prečno
validacijo (za vsak časovni korak i > x se izračuna
model nad podatki 1-i in napoved nad podatki i+1,
i+2, .. i+m) in beleženje napake RMSE, MAE in
razložene variance (explained variance)
• Implementacija najbolših predhodnih modelov za
napovedovanje porabe električne energije [1] (linearni
model in model naključnih gozdov)

Opravljeno delo in glavni rezultati
• Uporaba Tensorflow / Keras
• Implementacija 1D konvolucijske nevronske mreže z
oknom z fiksno dolžino [5]
• Implementacija rekurenčne nevronske mreže z
sigmoid (tanh) nivojem [5]
• Tako kot v [1], imata tudi na naši podatkovni množici
linearni model in model naključnih gozdov manjšo
napako kot prvotna konvolucijska in rekurenčna
nevronska mreža
• Nadgradnja v neopredeljeno polno-konvolucijsko
nevronsko mrežo z arhitekturo definirano v [2]
• Nadgradnja v rekurenčno LSTM nevronsko mrežo z
arhitekturo definirano v [6]

Predviden plan preostanka dela
• (15.6.2018) Testiranje in analiza rezultatov
nadgrajenih nevronskih mrež nad različnimi časovnimi
okvirji
• (1.7.2018) Implementacija, analiza in testiranje
ansambelskih operatorjev nad zgrajenimi nevronskimi
mrežami
• (15.7.2018) Analiza in eksperimentiranje s
hiperparametri posameznega modela
• (15.8.2018) Dokumentirani rezultati in metode vseh
zgrajenih modelov
• (01.09.2018) Končana prva verzija magistrske naloge
oddana mentorju v pregled
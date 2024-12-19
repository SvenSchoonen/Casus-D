# Opdracht 2 Casus D

## splitsing horizontaal (per patiënt een deel weghalen) of verticaal (een deel van de patiënten) 

### Horizontaal 

#### Nadel:

Er kan een risico op overfitting zijn als je model leert dat bepaalde patiënten altijd als geheel aanwezig zijn in de training of testset.


#### Voordelen:
Je hebt een afspiegeling van de beelden van elke patiënt in zowel de train- als testset.
Dit kan helpen bij de validatie van de generalisatie van je model voor nieuwe, niet-gezien patiënten.

Horizontaal splitsen betekent dat je per patiënt een bepaald percentage van hun beelden voor de testdata weghaalt,


### Vertical 

Verticaal splitsen betekent dat je de beelden van een bepaald aantal patiënten in de testset plaatst en de rest in de trainingsset. Dit kan helpen om ervoor te zorgen dat je model generaliseert naar patiënten die nog niet eerder in de trainingsset hebben gezeten, wat de performance van je model op onzichtbare data kan verbeteren.

#### Voordelen van verticaal splitsen:

Je zorgt ervoor dat je model niet "leert" van dezelfde patiënt tijdens zowel de training als de testfase, wat een meer realistische evaluatie van je model oplevert (vooral als je het model later op nieuwe, niet-gezien patiënten wilt toepassen).
Het voorkomt dat er te veel overfitting plaatsvindt, omdat je model wordt getest op patiënten die het niet eerder heeft gezien.
Nadelen van verticaal splitsen:

Het kan zijn dat de verdeling van gezonde en ongezonde gevallen niet gelijk is in de train- en testset. Dit kan worden opgelost door stratificatie (zorg ervoor dat de verhouding van de labels behouden blijft).

### Conclusie

Beide benaderingen hebben hun eigen voordelen, maar voor medische datasets is verticaal splitsen vaak nuttiger, omdat je model zo niet "leert" van dezelfde patiënt in zowel de trainings- als de testfase

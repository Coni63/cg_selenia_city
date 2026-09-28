# Selenia City - Sujet du problème

## Objectif

Le projet est maintenant officiel : Selenia City, la première ville lunaire, sera inaugurée en 2050 !

L'agencement définitif de la ville n'étant pas encore fixé, les architectes sont à la recherche d'une solution pouvant s'adapter au mieux à toutes les configurations possibles.

Suite à votre victoire aux jeux planétaires d'informatique, c'est donc tout naturellement que vous avez été appelé pour concevoir l'intelligence artificielle qui développera le réseau de transports de Selenia City.

---

## Règles

La partie se joue en **20** mois lunaires de **20** jours chacun.

Au début de chaque mois, vous recevrez des ressources ainsi que la liste des nouveaux bâtiments construits à Selenia City. Vous devrez alors utiliser judicieusement vos ressources pour raccorder les nouveaux bâtiments à votre réseau de transport ou renforcer les infrastructures existantes.

### Modes de transport

Deux modes de transport sont prévus sur la base lunaire : les **tubes magnétiques** et les **téléporteurs**.

<details>
<summary><b>Tubes magnétiques</b> (Cliquez pour afficher les détails)</summary>

![Déplacement d'une capsule](https://static.codingame.com/servlet/fileservlet?id=130026541439859)
*Déplacement d'une capsule*

Les tubes magnétiques sont construits sur le sol lunaire et permettent de transporter des passagers dans des capsules. Ces tubes sont construits en ligne droite entre deux bâtiments et sont bidirectionnels.

Deux tubes **ne peuvent pas se croiser**, et chaque bâtiment ne peut être relié qu'à **5** tubes au maximum. Un tube magnétique ne peut pas non plus traverser un bâtiment sans s'y arrêter.

Chaque tube ne peut initialement accueillir qu'une seule capsule à la fois, qui traverse celui-ci en une journée (quelle que soit sa longueur) en embarquant un maximum de **10** passagers. Il est néanmoins possible d'augmenter la capacité d'un tube en dépensant des ressources (voir détail plus bas).

![Un tube amélioré](https://static.codingame.com/servlet/fileservlet?id=130026566301239)
*Un tube amélioré*

Vous aurez la possibilité de configurer précisément l'itinéraire de chaque capsule, qui peut se déplacer librement dans le réseau de tubes magnétiques.

</details>

<details>
<summary><b>Téléporteurs</b> (Cliquez pour afficher les détails)</summary>

Les téléporteurs forment un lien de transport instantané d'un bâtiment vers un autre. Ils ont une capacité de passagers illimitée, et les trajets de deux téléporteurs peuvent se croiser.

![Un groupe d'astronautes en téléportation](https://static.codingame.com/servlet/fileservlet?id=130026520023803)
*Un groupe d'astronautes en téléportation*

Les équipements de réplication étant très volumineux, chaque bâtiment peut accueillir au maximum une entrée **ou** une sortie de téléporteur.

</details>

---

### Bâtiments

Les bâtiments de Selenia City sont organisés sur une zone rectangulaire de 160 x 90 kilomètres. Deux types de constructions existeront dans la ville : **les modules lunaires** et les **aires d'atterrissage**.

<details>
<summary><b>Modules lunaires</b> (Cliquez pour afficher les détails)</summary>

![Les 20 types de modules lunaires à Selenia City](https://static.codingame.com/servlet/fileservlet?id=130533301300005)
*Les 20 types de modules lunaires à Selenia City*

Il existe différents types de modules lunaires (laboratoire, site de prélèvements, observatoire, ...) qui abriteront les habitants de la ville.

Vous devrez transporter chaque astronaute vers un bâtiment du type correspondant pour gagner des points. Il peut exister plusieurs modules d'un même type.

</details>

<details>
<summary><b>Aires d'atterrissage</b> (Cliquez pour afficher les détails)</summary>

![Livraison de 20 astronautes sur une aire d'atterrissage](https://static.codingame.com/servlet/fileservlet?id=130026589572912)
*Livraison de 20 astronautes sur une aire d'atterrissage*

Les sites d'atterrisage sont des bâtiments au fonctionnement spécial : à chaque début de mois, une fusée déposera sur chacun d'entre eux un groupe d'astronautes. La composition du groupe d'astronautes reste identique chaque mois : si par exemple une aire d'atterrissage reçoit 5 techniciens de laboratoire et 10 employés d'observatoire, un groupe identique arrivera au début de chaque mois.

Si plusieurs modules lunaires ont le type recherché par un astronaute, celui-ci pourra se rendre vers n'importe lequel de ces bâtiments.

Votre solution marquera davantage de points si elle parvient à bien équilibrer la population entre les modules lunaires de même type.

</details>

---

### Calcul du score

Votre objectif est d'avoir un maximum de points à l'issue de la simulation.

Chaque astronaute qui atteint un bâtiment cible avant la fin du mois lunaire rapportera jusqu'à **100** points :

*   Pour la **rapidité** : **50** points, auxquels on soustrait le nombre de jours nécessaires à l'acheminement de l'astronaute.
*   Pour l'**équilibrage** de la population : **50** points, auxquels on soustrait le nombre d'astronautes déjà installés dans le module d'arrivée lors du mois lunaire en cours. S'il est négatif, ce score sera ramené à **0**.

<details>
<summary><b>Exemples</b> (Cliquez pour afficher les détails)</summary>

<u>Exemple 1 :</u> Un astronaute de type 4 arrive à Selenia City le premier jour du mois, et emprunte immédiatement un téléporteur qui l'amène de son aire d'atterrissage à un bâtiment de type 4.

Il est le premier à arriver sur sa base, vous gagnez donc **100 points** (50 points de rapidité et 50 points d'équilibrage).

<u>Exemple 2 :</u> Un astronaute emprunte un tube le premier jour pour quitter son aire d'atterrissage, puis un téléporteur suivi d'un tube le deuxième jour. Il arrive sur son module alors que 60 astronautes ont déjà été installés dans ce bâtiment depuis le début du mois.

Il vous rapportera ainsi **48 points** (48 points de rapidité et aucun point d'équilibrage).

</details>

---

## Implémentation

Chaque mois lunaire se déroule en 4 étapes :

### 1. Lecture de la carte

Au début de chaque mois, votre code recevra des informations concernant les nouvelles constructions à Selenia City.

<details>
<summary><b>Entrées du programme</b> (Cliquez pour afficher les détails)</summary>

*   **Sur la première ligne**, un entier `resources` représentant le nombre total de ressources à votre disposition.
*   **Sur la ligne suivante**, un entier `numTravelRoutes`, le nombre de chemins (tubes ou téléporteurs) actuellement présents sur la ville.
*   **Sur les `numTravelRoutes` lignes suivantes**, une description d'un tube ou d'un téléporteur sous la forme de 3 entiers `buildingId1`, `buildingId2` et `capacity` séparés par des espaces :
    *   `buildingId1` et `buildingId2` sont les deux extrémités du téléporteur ou du tube.
    *   `capacity` vaut **0** si la ligne est un téléporteur, et représente la capacité du tube sinon.
*   **Sur la ligne suivante**, un entier `numPods`, le nombre de capsules présentes dans le réseau.
*   **Sur les `numPods` lignes suivantes**, la description d'une capsule sous la forme d'une liste d'entiers séparés par des espaces :
    *   Le premier entier représente l'identifiant unique de la capsule de transport.
    *   Le deuxième entier indique le nombre `numStops` d'arrêts sur le trajet de la capsule.
    *   Les `numStops` entiers qui suivent représentent l'itinéraire de la capsule, soit les identifiants de tous les bâtiments sur son trajet.
*   **Sur la ligne suivante**, un entier `numNewBuildings`, le nombre de nouveaux bâtiments.
*   **Sur les `numNewBuildings` lignes suivantes**, la description d'un bâtiment venant d'être construit, sous la forme d'une liste d'entiers séparés par des espaces. Le format de chaque ligne dépend du type de bâtiment :
    *   Si le bâtiment est une aire d'atterrissage : `0 buildingId coordX coordY numAstronauts astronautType1 astronautType2 ...`
    *   Sinon, le premier nombre de la ligne est strictement positif et le bâtiment est un module lunaire : `moduleType buildingId coordX coordY`

Voici un exemple de données qui pourraient être fournies à votre programme au début d'un tour de jeu, avec une explication de chaque ligne :

![Exemple de données d'entrée](https://static.codingame.com/servlet/fileservlet?id=129813963720775)

</details>

---

### 2. Construction de l'infrastructure de transport

Votre code pourra ensuite faire évoluer l'infrastructure de transport par le biais de certaines actions de construction et d'amélioration.

<details>
<summary><b>Actions possibles</b> (Cliquez pour afficher les détails)</summary>

*   `TUBE buildingId1 buildingId2` : construction d'un tube magnétique entre deux bâtiments. Le coût est de **1** ressource pour chaque 0.1km de tube installé, arrondi à l'entier inférieur.
*   `UPGRADE buildingId1 buildingId2` : augmentation de la capacité d'un tube. Vous devrez dépenser le coût de construction initial multiplié par la nouvelle capacité. Par exemple, si un tube a coûté 500 ressources à construire, vous devrez dépenser 1000 ressources pour augmenter sa capacité à 2 capsules, puis 1500 ressources pour 3 capsules, etc.
*   `TELEPORT buildingIdEntrance buildingIdExit` : construction d'un téléporteur. Cette action coûte **5 000** ressources.
*   `POD podId buildingId1 buildingId2 buildingId3 ...` : création d'une capsule de transport et définition de son itinéraire. L'identifiant de la capsule doit être unique et compris entre **1** et **500**. Si le dernier bâtiment du trajet est égal au premier, la capsule parcourra son itinéraire en boucle, sinon elle s'arrêtera après avoir atteint le dernier arrêt. Cette action coûte **1 000** ressources.
*   `DESTROY podId` : déconstruction d'une capsule de transport. Cette action vous permet de récupérer **750** ressources.
*   `WAIT` : pour n'effectuer aucune action.

Pour exécuter plusieurs actions lors d'un même tour de jeu, vous pouvez séparer celles-ci avec un point-virgule comme suit : `TELEPORT 12 34;TUBE 23 45;UPGRADE 23 45`.

Si une action est impossible (par manque de ressources, ou si une construction de tube croise le trajet d'un autre tube par exemple), celle-ci sera ignorée et un avertissement s'affichera dans la console de jeu. Pour savoir quelles actions se sont déroulées avec succès, votre code peut utiliser les données d'entrée qui lui sont fournies au début du mois suivant.

Notez également que les tubes magnétiques et les téléporteurs sont permanents et ne peuvent pas être retirés une fois construits. Il est possible de changer le trajet d'une capsule en la détruisant et en la reconstruisant sur un autre itinéraire, ce qui vous coûtera 250 ressources.

</details>

---

### 3. Déplacement des astronautes

Une fois que vous avez terminé vos modifications à l'infrastructure de transport, 20 jours lunaires s'écoulent durant lesquels les astronautes se déplacent sur le réseau de manière autonome pour rejoindre leurs modules.

<details>
<summary><b>Simulation des déplacements</b> (Cliquez pour afficher les détails)</summary>

La traversée d'un tube magnétique prend toujours une journée, quelle que soit la distance parcourue. Sur un réseau donné, on peut donc définir la distance d'un bâtiment vers un autre comme le nombre minimal de tubes à emprunter pour effectuer le trajet complet (en s'autorisant aussi à utiliser les téléporteurs).

Les astronautes planifient leur trajet de manière naïve, en cherchant à se rapprocher du bâtiment cible le plus proche (ou de n'importe quel bâtiment cible le plus proche s'il en existe plusieurs). Ils emprunteront ainsi le premier tube ou téléporteur disponible qui les rapproche de leur destination, sans considérer les capsules des jours suivants pour déterminer si leur trajet est optimal ni même réalisable.

La phase de déplacement se déroule en 4 étapes chaque jour :

1.  **Téléporteurs** : chaque astronaute se trouvant sur l'entrée d'un téléporteur empruntera celui-ci, si le bâtiment d'arrivée a une distance **inférieure ou égale** à celle du bâtiment de départ. Comme la téléportation est instantanée, l'astronaute pourra ensuite emprunter un tube magnétique dans la même journée.
2.  **Allocation des capsules dans les tubes magnétiques** : il est possible que plusieurs capsules souhaitent emprunter un tube n'ayant pas la capacité suffisante pour toutes les accueillir. Dans ce cas, les capsules ayant le plus petit identifiant ont la priorité pour se déplacer, et les autres restent sur place pour la journée.
3.  **Allocation des astronautes dans les capsules** : chaque astronaute essaie de trouver une capsule avec au moins un siège libre qui le rapprochera de sa destination (soit une distance **strictement inférieure**), et s'installe à bord. Les astronautes choisissent chacun leur tour, par ordre croissant de l'identifiant de leur aire d'atterrissage. Si plusieurs choix permettent à un astronaute de s'approcher de sa destination, il choisira la capsule avec l'identifiant le plus petit.
4.  **Lancement des capsules** : chaque capsule se rend à sa destination avec ses astronautes à bord, puis tous les passagers descendent des capsules.

![Animation des déplacements](https://static.codingame.com/servlet/fileservlet?id=130533326556374)

Dans l'animation ci-dessus, prenez le temps de comprendre le mouvement de chaque astronaute, qui peut parfois être surprenant :

*   Le premier jour, les astronautes rouges prennent le téléporteur vers le module 6, et il peut sembler contre-intuitif de "s'éloigner" de leur module cible. Cependant, les bâtiments 4 et 6 sont tous les deux à une distance 3 du module 0, et les astronautes utiliseront toujours un téléporteur tant que celui-ci **n'augmente pas** la distance vers leur cible.
*   Les astronautes verts sur l'aire d'atterrissage 3 peuvent se déplacer vers le bâtiment 2 ou 4, qui sont tous les deux à distance 1 d'un module vert. Le premier jour, plusieurs astronautes empruntent la capsule de 3 à 4 à cause de leur algorithme de recherche de chemin naïf, et se retrouvent coincés là jusqu'à la fin du mois puisqu'aucune capsule n'ira de 4 vers 1.
*   Souvenez-vous que les astronautes emprunteront une capsule uniquement si elle diminue **strictement** la distance vers leur cible. Le dixième jour, il paraît surprenant que les astronautes se déplacent de 3 vers 4 alors que ces deux bâtiments semblent être à la même distance du module 6. Cependant, **la distance de 4 à 6 est en réalité 0** grâce à la présence d'un téléporteur, puisque les calculs de distance comptent uniquement le nombre minimum de **tubes** pour se déplacer d'un bâtiment à un autre.

</details>

---

### 4. Fin du mois lunaire

À la fin de chaque mois, tous les astronautes restants disparaissent de l'aire de jeu et les capsules retournent immédiatement à leur point de départ. Toutes les ressources que vous n'avez pas utilisées rapportent un intérêt de **10%** (arrondi à l'entier inférieur).

---

### Contraintes

*   Au cours de la partie, au maximum **150** bâtiments seront construits.
*   Chaque aire d'atterrissage accueillera entre **1** et **100** astronautes chaque mois.
*   Il y aura au maximum **1000** astronautes arrivant chaque mois.
*   Il est garanti qu'aucun bâtiment n'apparaîtra sur le trajet d'un tube existant.
*   Votre programme devra renvoyer sa liste d'actions en moins de **500** millisecondes à chaque tour (**1000** millisecondes au premier tour).
*   Il est garanti que chaque astronaute arrivant sur une aire d'atterrissage aura au moins 1 module du même type déjà construit.

> **Note :** Les validateurs cachés, qui permettent le calcul du score de votre solution, seront changés entre la fermeture du challenge et le calcul du score final pour éviter les solutions codées en dur. Il vous est néanmoins garanti que les validateurs cachés seront similaires aux tests visibles.

---

## Astuces pour bien démarrer

*   Commencez par programmer les capsules de transport de manière simple : des allers-retours sur une ligne devraient vous permettre d'obtenir de bons résultats.
*   L'identifiant de chaque capsule, que vous choisissez au moment de sa construction, sert également à établir l'ordre de priorité lors des déplacements. Vous pouvez l'utiliser pour désigner des navettes qui n'auront jamais à attendre qu'un tube se libère.
*   Il vous reste trop de ressources en fin de partie ? Créez des capsules et des téléporteurs pour accélérer le trajet de vos astronautes et marquer plus de points.
*   Si certains astronautes restent bloqués à cause de leur méthode naïve de recherche de chemin, essayez de faire en sorte que tous les tubes soient régulièrement traversés par des capsules dans chacun des deux sens.
*   Pour pouvoir créer un tube magnétique, celui-ci doit remplir 2 conditions : ne traverser aucun bâtiment (à l'exception de ses deux extrémités), et ne croiser aucun tube existant. Si vous avez du mal, n'hésitez pas à utiliser les deux algorithmes fournis ci-dessous :

<details>
<summary><b>Calcul d'intersection point-segment</b> (Cliquez pour afficher les détails)</summary>

Pour vérifier si un bâtiment A se trouve sur un segment BC, on peut créer le triangle ABC et vérifier si la distance BC est égale à BA+AC.

```javascript
fonction distance(p1, p2) {
    renvoyer sqrt((p2.x-p1.x)² + (p2.y - p1.y)²)
}

fonction pointOnSegment(A, B, C) {
    epsilon = 0.0000001
    renvoyer (-epsilon < distance(B, A) + distance(A, C) - distance(B, C) < epsilon)
}
```

Ici, la présence de `epsilon` sert à comparer des nombres flottants malgré leur précision limitée (dans la plupart des langages, `0.1 + 0.2 != 0.3`). Il existe d'autres méthodes utilisant uniquement des entiers, mais il vous est garanti que la fonction ci-dessus est valide pour tous les points entiers de la grille 160x90 avec la valeur d'epsilon donnée.

</details>

<details>
<summary><b>Calcul d'intersection segment-segment</b> (Cliquez pour afficher les détails)</summary>

Pour vérifier si deux segments AB et CD se croisent en dehors de leurs extrémités, vous pouvez utiliser la fonction `segmentsIntersect` définie ci-dessous :

```javascript
fonction orientation(p1, p2, p3) {
    prod = (p3.y-p1.y) * (p2.x-p1.x) - (p2.y-p1.y) * (p3.x-p1.x)
    renvoyer sign(prod)
}

fonction segmentsIntersect(A, B, C, D) {
    renvoyer orientation(A, B, C) * orientation(A, B, D) < 0 && orientation(C, D, A) * orientation(C, D, B) < 0
}
```

Ici, la fonction `sign(x)` renvoie -1 si x est négatif, 0 s'il est nul et 1 s'il est positif.

</details>

### 🐞 Astuces de débogage

*   Passez votre souris sur un tube, une capsule ou un bâtiment pour voir davantage d'informations à son propos.
*   Cliquez sur l'engrenage dans l'interface de jeu pour accéder à des options d'affichage supplémentaires.
*   Utilisez le clavier pour contrôler l'affichage : barre espace pour play/pause, et les flèches pour faire défiler chaque jour.

---

### Game source

The CodinGame game source lives in `FallChallenge2024-SeleniaCity/` (cloned from
https://github.com/0x6E0FF/FallChallenge2024-SeleniaCity). It includes a custom
`com.codingame.bench.BenchRunner` (`src/main/java/com/codingame/bench/BenchRunner.java`)
and a `maven-shade-plugin` config in `pom.xml` that together produce a standalone
benchmark jar — no CodinGame IDE/export needed.

Build it with:

```sh
cd FallChallenge2024-SeleniaCity
mvn -q -DskipTests package
```

This produces `FallChallenge2024-SeleniaCity/target/fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar`,
which runs every `testN.json` in `FallChallenge2024-SeleniaCity/config/` against an
agent command and reports the score (parsed from the referee's `points` metadata) for each.

### Python solution dependencies

```sh
cd python_solution
poetry install --no-root
```

### Running the benchmark

`bench_full.py` (at the repo root) wraps the jar: it builds it if missing, picks the
`python_solution/.venv` interpreter by default, and runs the full test suite.

```sh
python bench_full.py                # full 24-test benchmark
python bench_full.py --test 8       # single test case
python bench_full.py --rebuild      # force a fresh mvn package before running
```

To benchmark a different solver (e.g. a future Rust port), pass `--solution`:

```sh
python bench_full.py --solution rust_solution/target/release/agent.exe
```

Under the hood this is equivalent to:

```sh
java -jar FallChallenge2024-SeleniaCity/target/fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar \
     "<python_solution/.venv python> python_solution/main.py" \
     FallChallenge2024-SeleniaCity/config ref_scores.txt [testNumber]
```

Set `BENCH_VERBOSE=1` to print each test's stderr/summary output for debugging.

### Sources

- https://www.codingame.com/forum/t/fall-challenge-2024-feedback-and-strategies/205205/11

- https://github.com/mourner/delaunator-rs

- https://virtual-atom.com/codingame/fall24/

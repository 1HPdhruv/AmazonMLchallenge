# Error analysis (TEST split, final system)

**SYNTHETIC VALIDATION — NOT COMPETITION PERFORMANCE**

## Diagnostic tree
- union candidate recall on evaluated true pairs: 0.9823 (target 0.95)
- entities with a retrieved true match where best-true > best-false score: 0.9949 (n=790)
- retrieval_miss: 19 (13.9% of failures)
- model_miss: 3 (2.2% of failures)
- decision_miss: 33 (24.1% of failures)
- no_match_false_positive: 7 (5.1% of failures)
- multi_under_selection: 66 (48.2% of failures)
- over_selection: 9 (6.6% of failures)

## Error class by stratum

```
cand_bucket               20+  5-20
error_class                        
correct                  1019    44
decision_miss              32     1
model_miss                  3     0
multi_under_selection      65     1
no_match_false_positive     7     0
over_selection              9     0
retrieval_miss             18     1
```

```
country                   BR   DE   FR   IN  JP   US
error_class                                         
correct                  101  149  131  155  94  433
decision_miss              3    2    6    3   2   17
model_miss                 1    0    0    1   0    1
multi_under_selection     12    9    4    8   4   29
no_match_false_positive    0    0    1    1   0    5
over_selection             1    2    2    2   0    2
retrieval_miss             2    1    2    1   2   11
```

```
true_sources             none  source2  source2+source3  source3
error_class                                                     
correct                   392      333              120      218
decision_miss               0       12                0       21
model_miss                  0        1                0        2
multi_under_selection       0        7               53        6
no_match_false_positive     7        0                0        0
over_selection              0        6                2        1
retrieval_miss              0        8                5        6
```

### Highest-confidence accepted matches

| entity | record | score | label | S1 name / addr | record name / addr |
|---|---|---|---|---|---|
| E0005988 | S2_0005398 | 1.000 | 1 | Harbor Eagle Imports LLC / 21 hill avenue houston 46485 | Eagle Harbor Imports LLC / 21 hill ave houston 46485 |
| E0000006 | S3_0000006 | 1.000 | 1 | Orzelcorka Software Inc / 1074 elm avenue houston 27642 | Orzeclorka Software / 1074 elm avenue houston 27642 |
| E0000006 | S3_0000007 | 1.000 | 1 | Orzelcorka Software Inc / 1074 elm avenue houston 27642 | Orzeliorka  Software, / 1074 elm ave houston |
| E0000009 | S2_0000013 | 1.000 | 1 | Marbelrin Imports Inc / 3308 rosen boulevard houston 27642 | Marbelrin Imporis Inc / 3308 rosen boulevard houston 27642 |
| E0000009 | S2_0000014 | 1.000 | 1 | Marbelrin Imports Inc / 3308 rosen boulevard houston 27642 | MARBELRIN IMPORTS INC / 3308 rosen blvd houston 27642 |
| E0000016 | S2_0000017 | 1.000 | 1 | Royal North Systems AG / sunset weg 110 frankfurt 57994 | ROYAL NORTH SYSETMS / sunset weg 110 57994 |
| E0000019 | S3_0000013 | 1.000 | 1 | Vexvoor Foods AG / gardenstrasse 195 munchen 66781 | Vexvoor  oFods AG, / gardenstr 195 66781 munchen |
| E0005983 | S3_0004263 | 1.000 | 1 | First Vexbelcas Bakery Co / 152 main avenue suite 80 denver 46326 | fairst vexbelcas bakery / 152 main avenue 80 denner 46326 |

### Lowest-confidence accepted matches

| entity | record | score | label | S1 name / addr | record name / addr |
|---|---|---|---|---|---|
| E0005277 | S2_0004767 | 0.692 | 1 | Silver Lumkorsta Pharma Inc / 4009 pine street houston 22193 | Silver Lumkorsta Pharma / 4011 pine st houston 22193 |
| E0002325 | S3_0001739 | 0.692 | 1 | Doryorka Media Private Limited / 248 rosen road chennai 647689 | Doyriorka  Pvt, / 248 rosen road chennai |
| E0002994 | S3_0002182 | 0.692 | 1 | North Stavexlanwyn Motors Co Ltd / 3050 6 ginza machi tokyo 294 1399 | NSMC Ltd / 3050 6 ginza machi tokyo 294 1399 |
| E0004026 | S3_0002916 | 0.692 | 1 | Harbor Nexvocor Pharma Group LLC / seattle 38925 | Harbor Nexvocor Phamra uGroup / 233 king boulevard wa 38925 |
| E0002132 | S2_0001927 | 0.692 | 1 | Korfalliami Motors Pvt Ltd / 4 market marg delhi 620574 | Motors Kofalliamu Pvt / 4 markeo marg delhi |
| E0001329 | S2_0001223 | 0.692 | 1 | The Yorlanlizel Trading Enterprises ME / rua goethe 19 recife 37282 310 | The Yrlanlizel Trading Ent ME / r goethe 17 recife |
| E0001568 | S2_0001430 | 0.695 | 1 | North Stavexlanwyn Motors KK / 19 6 sakura machi osaka 262 2318 | North  Stavexlanwyn Motors KK, /  |
| E0004759 | S3_0003426 | 0.700 | 0 | Paxkaor Coffee Co / 292 park street suite 110 houston 97974 | PAXKOAR COFFEE CO / 322 park st 110 houston 97974 |

### False positives (accepted, label 0)

| entity | record | score | label | S1 name / addr | record name / addr |
|---|---|---|---|---|---|
| E0003031 | S2_0002746 | 1.000 | 0 | Marbrihel Construction GmbH / church weg 44 munchen 10221 | MARBRIREL CONSTR GMBH / church weg 45 munchen 10221 |
| E0003751 | S3_0002709 | 0.900 | 0 | The Corhelli Bakery Ltd / 294 station marg delhi 309030 | None / 294 station marg delhi |
| E0005409 | S2_0004897 | 0.900 | 0 | Star Druzel Furniture Co / 17 lincoln avenue suite 240 boston 57511 | Star Druzel Furnitre / 27 lincoln ave ste 240 57511 |
| E0001056 | S3_0000766 | 0.875 | 0 | Summit Rinxaeli Construction SA / avenida bridge 183 recife 89410 732 | Summit Rinxaeli Constr SA / avenida bridge 185 recife 89410 732 |
| E0003720 | S2_0003343 | 0.857 | 0 | The Bridortor Bäckerei Systems AG / mainstrasse 238 frankfurt 15874 | THE BRIDORTOR BACKEREIS SYSTEMS AG /  |
| E0004103 | S2_0003673 | 0.857 | 0 | Helxa Société Bakery SARL / 2286 boulevard de nehru lyon 13084 | Helxa Société Bakery / 2309 bd de nehru lyon |
| E0000507 | S3_0000378 | 0.857 | 0 | North Rarinzerin Foods Corp / 296 main avenue austin 52029 | North Rarinzerin Foods Corp /  |
| E0000233 | S2_0000228 | 0.857 | 0 | The Dororfincas Dental Company Inc / 3 park avenue houston 63501 | The Dororfincas Dentl Company Inc /  |

### False negatives among retrieved (rejected, label 1)

| entity | record | score | label | S1 name / addr | record name / addr |
|---|---|---|---|---|---|
| E0000128 | S3_0000091 | 0.722 | 1 | Summit Atlas Hotels SA / rua sakura 1 rio de janeiro 91780 145 | Summit Atlas Hotels SA / rua sakura i 91780 145 |
| E0000495 | S2_0000470 | 0.722 | 1 | City Drufinor Solutions SA / travessa bridge 133 curitiba 53095 850 | None / tv bridge 133 curitiba |
| E0001072 | S3_0000785 | 0.722 | 1 | Blue Vista Comércio Foods ME / travessa hill 215 salvador 82743 539 | BLEU VISTA FOODS COMÉRCIO ME /  |
| E0005667 | S3_0004061 | 0.722 | 1 | Amizeldorhel Hotels Ltda / avenida park 9 sao paulo | Amizeldorhel  Hotels Ltda. /  |
| E0005815 | S3_0004156 | 0.722 | 1 | Ulstasta Hotels Corp / 29 nehru boulevard suite 290 austin 17568 | USTASTA  CORP HOTELS. /  |
| E0004595 | S3_0003308 | 0.722 | 1 | City Apex Engineering Enterprises LLC / 229 sakura avenue chicago 71743 | City Apex Engineering Enterprises /  |
| E0005165 | S3_0003709 | 0.722 | 1 | Bricasbel Software Enterprises Private Limited / 28 ginza road delhi | BSEP Limited / 28 ginza road delhi 303404 |
| E0003600 | S3_0002600 | 0.722 | 1 | City Paxfaltribel Foods Holdings Co Ltd / 8591 3 paix machi nagoya 688 3324 | city paxfaltribel foods holdings co ltd /  |

### No-match entities: best candidate

| entity | record | score | label | S1 name / addr | record name / addr |
|---|---|---|---|---|---|
| E0000507 | S3_0000378 | 0.857 | 0 | North Rarinzerin Foods Corp / 296 main avenue austin 52029 | North Rarinzerin Foods Corp /  |
| E0002962 | S3_0002701 | 0.857 | 0 | Silver Lumkorsta Pharma Inc / 117 mozart road miami 64542 | None / 117 mozart road 240 miami 64542 |
| E0000233 | S2_0000228 | 0.857 | 0 | The Dororfincas Dental Company Inc / 3 park avenue houston 63501 | The Dororfincas Dentl Company Inc /  |
| E0003286 | S3_0002390 | 0.768 | 0 | Korraami Société Systems International SARL / nantes 41282 | Korraami  Systems Société International SARL. / 27 rue de mozart marseille |
| E0003814 | S3_0002754 | 0.722 | 0 | Lumcaslum Systems Enterprises Corp / new york 12160 | LUMCASLUM SYSTEMS ENTERPRISES CORP /  |
| E0004071 | S2_0003643 | 0.700 | 0 | Alpha Lizeleli Coffee Ltd / 5050 liberdade road bengaluru 405908 | Arpha Lizeleli Coffee Ltd / 5062 liberdade road bengaluru 405908 |
| E0004759 | S3_0003426 | 0.700 | 0 | Paxkaor Coffee Co / 292 park street suite 110 houston 97974 | PAXKOAR COFFEE CO / 322 park st 110 houston 97974 |
| E0003494 | S2_0003119 | 0.375 | 0 | Golden Nogal Consulting LLC / 254 garden avenue suite 310 seattle 63561 | Golden Nogal Consulting LLC / 279 garden avenue seattle 63561 |

### Retrieval misses (true pair never retrieved)

| entity | S1 name / addr | record name / addr |
|---|---|---|
| E0002500 | City Libristapax Media LLC / 18 linden avenue chicago 34172 | City rLibristapax LLC / 19 linde navenue chicago |
| E0004485 | Star Coffee Co / 24 victoria avenue miami 69887 | StarCoffee. / 24 virtoria ave fl |
| E0003341 | Liberty Druqumaren Foods Co / 11 lake avenue boston 84791 | None /  |
| E0001436 | Yormarmar Software LLC / 29 king avenue denver 77445 | None /  |
| E0000631 | Stafal Coffee GmbH / park weg 132 munchen 10221 | None /  |
| E0002994 | North Stavexlanwyn Motors Co Ltd / 3050 6 ginza machi tokyo 294 1399 | NORTH STAVEXLANWYN MOTORS CO STD /  |
| E0000658 | Green Quamicor Logistics SA / 9801 rue de main lyon 13084 | None /  |
| E0000953 | Falcorelinex Construction Co / austin 36684 | None / 11 oak blvd suite 320 |
| E0004468 | Liberty Xavexami Logistics ME / avenida mill 166 salvador 82743 539 | LIBERTY XAVNXAMI LOGISTICS ME /  |
| E0003527 | Bright Blue Bakery KK / 165 5 fleurs dori nagoya 384 3924 | brigth ble bakery / 165 5 fleus dori |
| E0004145 | Green Ringalra Auto Parts Pvt Ltd / hyderabad 112962 | GRAPP Ltd /  |
| E0000752 | Eagle Lumheltriqu Furniture Corp / 102 bridge boulevard new york 12160 | None /  |

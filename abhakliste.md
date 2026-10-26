# BiQwen: Korrektur und vollständiges Training

Stand: 26. Oktober 2026. Grundlage ist die Feasibility-Prüfung des Commits
`044cf99` auf `experiment/biqwen-extract-decide-4b`.

Ziel ist ein generalistischer DE/EN-Extraktor mit Qwen-Backbone, GLiNER-artigen
Extraktionsköpfen und der bestehenden programmatischen Normalisierung. Maßgeblich
sind die endgültigen Antworten der einzigen Tractor-API, einschließlich `issues`.

Diese Liste verfolgt die Umsetzung einschließlich noch offener Arbeit. Ein Haken bedeutet, dass der
Schritt ausgeführt und sein Ergebnis nachgewiesen wurde. Vorhandene Vorbereitung,
Softwaretests, GPU-Prüfung, abgeschlossenes Training und erreichte Modellqualität
sind getrennte Zustände.

## Ausgangsproblem und Korrekturstrategie

Die geprüfte Architektur besteht aus einem kausalen Qwen-Backbone, einer
Projektion von 2.560 auf 512 Dimensionen, zwei bidirektionalen Adapter-Schichten
und GLiNER-artigen Extraktionsköpfen. Anschließend verarbeitet die bestehende
Tractor-Pipeline die extrahierten Werte. Das ist technisch eine mögliche
Extraktionsarchitektur; der gesamte Qwen-Backbone wird dadurch nicht bidirektional.

| Befund der Machbarkeitsprüfung | Geplante Korrektur | Verbindlicher Nachweis |
| --- | --- | --- |
| Feldtypen beeinflussen bisher nur die Normalisierung, nicht unmittelbar den Modelleingang. | Gemeinsames typisiertes Schema-Encoding für Compiler und Inferenz (Abschnitt 1). | Unterschiedliche Typen erzeugen unterschiedliche Eingaben; Training und API verwenden identische Eingaben. |
| Der additive Record-Scorer kann die Besitzerpräferenz einer Fundstelle nicht feldabhängig umkehren. | Gemeinsame nichtlineare Verarbeitung von Anker und Feld (Abschnitt 2). | Ausdrucksfähigkeit, Gradienten und Checkpoint-Roundtrip prüfen; den Qualitätsgewinn anschließend auf vollständigem Dev messen. |
| Gesamt-Loss verdeckt fehlende Vorschläge und ausgelassene Zuordnungsziele. | Komponentenverluste und Kandidatenabdeckung dauerhaft erfassen (Abschnitt 3). | Abdeckung vor Gold-Injektion und endgültige Pipelinefehler getrennt ausweisen. |
| Vollständige Datenabdeckung bedeutet keine ausgewogenen Fähigkeiten. | Alle geeigneten Daten erhalten; Sprache, Domäne und Ausgabeform getrennt bewerten (Abschnitte 4, 8 und 9). | Keine versteckten Quoten; eventuelle Loss-Gewichtung ausdrücklich konfigurieren und auf Dev prüfen. |
| CPU-Tests belegen nicht die Speicherverträglichkeit des tatsächlichen 4B-Trainings. | Echte GPU-Forward/Backward-Prüfung vor dem Volltraining (Abschnitte 6 und 7). | Unterstützte Eingabegrenzen, endliche Gradienten und unveränderte Gewichte ohne Optimizer-Updates nachweisen. |
| Die Kapazität des bidirektionalen Adapters ist als Fehlerursache bislang nicht isoliert. | Backbone zunächst erhalten; einen Umbau nur aus verbleibenden Fehlern ableiten (Abschnitt 9). | Keine Kapazitätsbehauptung allein aus Gesamt-F1 oder bestandenen Softwaretests ableiten. |

Die Abschnitte 0 bis 7 schaffen die Startbedingungen, Abschnitt 8 führt das
vollständige Training durch und Abschnitt 9 prüft die tatsächliche Verbesserung.
Ein technischer Nachweis ersetzt dabei keinen Qualitätsnachweis.

## Aktueller Nachweisstand

- Architektur und Typ-Encoding umgesetzt; vollständige Baselines verglichen.
- Vollständiger finaler Korpus: 339.949 Dokumente / 342.147 Requests. Alle Texte,
  Spannen und Bindungen wurden gegen den vorherigen vollständigen Korpus geprüft.
- Korpusmanifest: `849ecce3bac822494cd3d9e0442a61b21c3c28371aa8c7cfc1bba4a5b65db452`.
- Nachweise: `reports/biqwen-correction-20261026/`, insbesondere
  `corpus-audit-final.json`, `legacy-baseline/verification.json`,
  `gpu-ready/report.json` und die Prüfprotokolle.
- 891 Softwaretests bestanden; sieben echte GPU-Forward/Backward-Fälle bestanden,
  ohne Gewichtsupdates. Der vollständige Trainings-Preflight ist bestanden.
- Softwareprüfungen sind keine Qualitätsmessung. Das Volltraining läuft; vollständige Epochen, Modellauswahl und Abschnitt 9
  sind noch nicht abgeschlossen.
- Die Nachbearbeitung für Abschnitt 9 ist separat vorbereitet: Commit `1d906ef`
  im Worktree `/scratch/johann-peter.hartmann/tractor-biqwen-v2/post-training`.
  913 Softwaretests sowie Lint und Dokumentationsprüfung bestehen. Enthalten sind
  die Sperre vor vollständigem Training/Dev-Auswahl, Kalibrierung über die
  Produktionspipeline, ein gebundenes Inferenz-Bundle und der eingefrorene
  Abschlusstest. Der Record-Replay wurde mit echten parametrisierten Köpfen und
  PyTorch-Inferenztensoren geprüft. Reale Kalibrierung und Test sind noch offen.
  Diesen Commit erst nach Ende des Trainings übernehmen: Der laufende Trainer
  prüft die unveränderten Hashes seines ursprünglichen Implementierungsstands.

## Verbindliche Leitplanken

- [ ] Die Vorgaben aus [AGENTS.md](AGENTS.md) und [reqs.md](reqs.md) für die
  Umsetzung berücksichtigen und bei jeder Änderung am Scope prüfen.
- [ ] Genau eine öffentliche API erhalten: `POST /v1/extract` und `Client.extract`;
  Antwort ausschließlich mit `data` und `issues`.
- [ ] Einfache Pflicht-Feldtypen und deren Normalisierungsregeln erhalten;
  keine zusätzliche öffentliche Schema- oder Verarbeitungs-API einführen.
- [ ] Nur ausdrücklich im Text stehende Werte, Entitäten und unterstützte
  wiederholte Datensätze aus lesbarem Text trainieren.
- [ ] CORD in allen Splits ausgeschlossen halten. Ebenso ausgeschlossen bleiben
  Klassifikation, Routing, Sentiment, Dringlichkeit, freie QA, generische
  Relationsextraktion, Reasoning, OCR und Tabellen-/Positionsrekonstruktion.
- [ ] Sämtliche geeigneten deutschen und englischen Trainingsdaten verwenden;
  keine versteckten Stichproben, Domänenquoten oder begrenzten Reparaturläufe.
- [ ] Keine Trainingsziele aus Modellvorhersagen oder Heldout-Antworten ableiten.
- [ ] Bestehende Normalisierungsbibliotheken, Decimal-Arithmetik, explizite
  Locale-/Ländervorgaben und Mehrdeutigkeitsbehandlung erhalten.
- [ ] Keine GPU-Latenzmessung, kein Cloud-Training und keine Veröffentlichung
  als Nebenwirkung dieser Arbeit ausführen.

## 0. Ausgangszustand und Nachweise sichern

- [x] Aktuellen Commit, Arbeitsbaumzustand und aktive Trainingskonfiguration
  im neuen Arbeitsbericht festhalten.
- [x] Die Befunde aus
  `reports/biqwen-feasibility-20261026/review.json` und
  `reports/biqwen-feasibility-20261026/structural-checks.json` übernehmen.
- [x] Pinned BiQwen-Bundle einschließlich Code, Tokenizer und Gewichten anhand
  `src/tractor/biqwen-lock.json` verifizieren.
- [x] Das vorhandene Bundle unverändert lassen; neue Architekturbausteine im
  versionierten Tractor-Code implementieren.
- [x] Den bereits zusammengeführten BiQwen-Backbone als Initialisierung binden.
  Neue LoRA-Adapter dürfen nicht auf dem ursprünglichen Qwen-Basismodell geladen
  werden, weil dadurch das bisherige BiQwen-Lernen verloren ginge.
- [x] Bisherige Baseline, Qualitätsregeln und Auswertungsvertrag sichern.
- [x] Neue Versionskennungen für Schema-Encoding, Architektur, native Daten und
  Trainingsrezept festlegen; inkompatible Kombinationen ausdrücklich ablehnen.
- [x] Den bisherigen vorbereiteten Lauf bis zum Abschluss der Startbedingungen
  aus Abschnitt 7 als überarbeitungsbedürftig kennzeichnen. Dabei keine
  laufenden Prozesse ohne Prüfung verändern.

**Abnahme:** Ausgangsmodell und Vergleichsdaten sind eindeutig identifizierbar;
der alte vorbereitete Lauf kann nicht versehentlich als korrigierter Lauf starten.

## 1. Feldtypen im Modelleingang verankern

Betroffene Stellen: insbesondere `src/tractor/biqwen.py`,
`src/tractor/biqwen_training_data.py` und die zugehörigen Tests.

- [x] Einen gemeinsamen, deterministischen Schema-Encoder für Trainingscompiler
  und Inferenz definieren.
- [x] Pro Feld die unveränderte fachliche Beschreibung und den einfachen
  API-Typ kodieren, beispielsweise `person`, `place`, `address`, `quantity`.
- [x] Für alle unterstützten Typen dokumentieren, welche Parameter für die
  Quellwertauswahl relevant sind und welche ausschließlich die Verarbeitung
  beziehungsweise Zielrepräsentation betreffen.
- [x] Relevante Parameter eindeutig kennzeichnen. Eine Ziel-Einheit `mm` darf
  ausdrücklich genannte Quellen wie `12 cm` nicht als unerwünscht darstellen.
- [x] Fehlende Länder, Währungen oder Datumsreihenfolgen nicht ergänzen oder
  aus dem Dokument erraten.
- [x] Anonyme Feld-IDs, Feldreihenfolge, Ankerbindung und Request-Grenzen
  zwischen Compiler und API identisch erhalten.
- [x] Die frühere GLiNER-spezifische Mengenbeschreibung nicht pauschal
  wiederherstellen; das neue Format gilt konsistent für alle Typen.
- [x] Prüfen, dass bei sonst identischem Request `person` und `place`
  unterschiedliche Modelleingaben erzeugen.
- [x] Prüfen, dass Training und API für denselben Request dieselben Token-IDs,
  Masken, Feld-Readouts und Dokument-Offsets erzeugen.
- [x] DE/EN-Beispiele mit Namen, Orten, Adressen, Datum, Telefonnummern,
  Kennungen und Mengen einschließlich abweichender Quell-/Zieleinheiten prüfen.
- [x] Nachweisen, dass die Kodierung ohne Goldwerte oder Offline-Annotationen
  vollständig erzeugt werden kann.

**Abnahme:** Der Typ kann die Extraktion beeinflussen, während Normalisierung und
Validierung weiterhin durch die bestehenden Regeln erfolgen. Es gibt keinen
Unterschied zwischen Trainings- und Inferenzformat.

## 2. Record-Zuordnung feldabhängig machen

Der aktuelle additive Scorer kann bei festen Span- und Ankerrepräsentationen die
Besitzerpräferenz derselben Fundstelle innerhalb eines Schemas nicht abhängig vom
Feld umkehren. Das ist eine nachgewiesene Ausdrucksgrenze, aber noch kein Nachweis
für die Ursache jedes beobachteten Record-Fehlers.

- [x] Einen Scorer implementieren, der Anker und Feld gemeinsam nichtlinear
  verarbeitet, beispielsweise als residuale Erweiterung der bisherigen Bewertung.
- [x] Die Initialisierung so gestalten, dass die vorhandenen Bewertungen
  zunächst erhalten bleiben und die neuen Parameter trotzdem lernen können.
- [x] Source-Score, Abwesenheitsalternative und Zuordnungsentscheidung in einer
  gemeinsamen Definition für Training und Decoder zusammenführen.
- [x] Die bestehenden positiven Mehrfachziele pro Datensatz und Feld erhalten:
  wiederholte Belege, unterschiedliche gültige Werte und explizite Konflikte.
- [x] Kandidaten anderer Datensätze als negative Zuordnungen berücksichtigen;
  fehlende Felder mit der vorgesehenen Abwesenheitsbehandlung trainieren.
- [x] Bei fehlenden Gold-Kandidaten keine falsche Abwesenheit lehren. Ausgelassene
  Zuordnungsziele zählen und die Ursache diagnostizieren.
- [x] Exklusive Zuordnung einer Fundstelle erhalten. Das Wiederverwenden einer
  einzelnen Erwähnung für mehrere Datensätze bleibt außerhalb des unterstützten
  Vertrags; separate Erwähnungen desselben Werts bleiben zulässig.
- [x] Mehrdeutige Zuordnungen nicht stillschweigend durch Textnähe, Listenposition
  oder einen anderen unangefragten heuristischen Ersatz entscheiden.
- [x] Einen gezielten Ausdrucksfähigkeitstest ergänzen: Bei festen Anker- und
  Kandidatenrepräsentationen müssen verschiedene Feldvektoren unterschiedliche
  Besitzerpräferenzen ausdrücken können.
- [x] Endliche, wirksame Gradienten für den neuen Scorer, die bestehenden Köpfe
  und die trainierbaren Backbone-Adapter nachweisen.
- [x] Tests für mehrere positive Fundstellen, falschen Besitzer, fehlendes Feld,
  konkurrierende Werte, gleichwertige Wiederholungen und Zuordnungsgleichstand
  durchführen.
- [x] Neue Parameter und Architekturversion vollständig im Checkpoint speichern;
  identische Ausgaben nach erneutem Laden prüfen.
- [x] Alte Checkpoints ausdrücklich über ihren alten Architekturpfad laden;
  inkompatible Checkpoints nicht stillschweigend mit fehlenden Parametern laden.

**Abnahme:** Die mathematische Einschränkung ist aufgehoben. Mehrfachwerte und
Issues bleiben vertragskonform. Ein bestandener Ausdrucksfähigkeitstest wird
nicht als Qualitätsverbesserung auf realen Dokumenten ausgegeben.

## 3. Lernfortschritt und Fehlerursachen sichtbar machen

- [x] Boundary-, Source- und Assignment-Loss getrennt mit Zähler und Nenner
  protokollieren, zusätzlich zum Gesamt-Loss.
- [x] Kandidatenabdeckung vor Gold-Injektion erfassen: Wie viele gültige
  Ziel-Spans sind durch die tatsächliche Vorschlagsauswahl erreichbar?
- [x] Gold-Injektionsrate, tatsächlich injizierte Spans, fehlende Anker und wegen
  fehlender Kandidaten ausgelassene Zuordnungsziele dauerhaft protokollieren.
- [x] Diagnostik nach Sprache, Domäne und Ausgabeform aggregieren, soweit diese
  Zuordnung durch die vorhandenen Metadaten belegt ist.
- [x] Kapazitätsüberschreitungen für Kontext, Span-Länge, Kandidaten und Records
  separat ausweisen.
- [x] Bei der Dev-Auswertung Gold-Injektion vollständig ausschalten.
- [ ] Endgültige Fehler in fehlenden Vorschlag, verworfenen Kandidaten, falsche
  Zuordnung, Normalisierungs-/Validierungsproblem und Referenzlücke aufschlüsseln.
- [x] Rohspan-Diagnostik von der Produktbewertung trennen: Zulässige inhaltliche
  Alternativen dürfen nicht erneut durch exakte Span-Gleichheit blockieren.
- [x] Vorhandene Offline-Antwortalternativen und unvollständige Referenzen
  korrekt behandeln; keine neue Fuzzy- oder Substring-Gleichsetzung einführen.
- [x] Bei Wiederaufnahme des Trainings fortlaufende und eindeutige Metrikstände
  erhalten, statt verworfene Updates doppelt zu zählen.

**Abnahme:** Ein sinkender Gesamt-Loss kann fehlende Kandidaten oder untrainierte
Zuordnungen nicht verdecken. Die finale Pipeline bleibt die Auswahlgrundlage.

## 4. Vollständigen Trainingskorpus neu kompilieren

Ausgangspunkt ist `data/full-training-de-en` mit 341.429 Dokumenten und 343.694
Tasks. Der bisherige Compiler akzeptierte 339.949 Dokumente beziehungsweise
342.147 Requests. Diese Zahlen sind Vergleichswerte, keine ungeprüfte Zusage für
das neue Encoding.

- [x] Quellmanifest, Quellenqualifikation und Split-Zuordnung verifizieren.
- [x] Alle deutschen und englischen Quelldokumente erneut durch den neuen
  gemeinsamen Schema-Encoder kompilieren.
- [x] Für jeden ursprünglichen Task genau einen Status ausweisen: übernommen
  oder mit einem konkreten technischen beziehungsweise fachlichen Grund verworfen.
- [x] Annahme und Ausschluss getrennt auf Dokument- und Task-Ebene zählen;
  teilweise nutzbare Dokumente nicht vollständig verlieren.
- [x] Offsets, Token-Grenzen, Feldbindungen und Record-Zuordnungen prüfen.
- [x] Zusätzliche Kontextlänge durch Typinformationen berücksichtigen; keinen
  Text, kein Feld und keinen Goldwert stillschweigend abschneiden.
- [x] Veränderungen gegenüber den bisher 1.547 ausgeschlossenen Trainings-Tasks
  nach Gründen und betroffenen Quellen ausweisen.
- [x] Nicht tokenkompatible Spans nicht automatisch verbreitern. Gültige
  Alternativen nur auf Grundlage des bestehenden Feldvertrags beziehungsweise
  einer nachvollziehbaren Quellenkorrektur übernehmen.
- [x] CORD und alle anderen ausgeschlossenen Aufgaben erneut über alle
  relevanten Datenartefakte prüfen.
- [x] Überschneidungen mit Dev, Kalibrierung und Test anhand der vorhandenen
  IDs, Gruppen und Text-Fingerprints ausschließen.
- [x] Native Requests und Identitätsdateien positionsgleich erzeugen und prüfen.
- [x] Vollständiges Dev mit 4.429 Dokumenten für die Pipeline-Auswertung erhalten;
  technische Ausschlüsse aus nativen Dev-Zielen dürfen keine Dokumente aus der
  Produktbewertung entfernen.
- [x] Sprach-, Quellen-, Domänen- und Ausgabeformverteilung dokumentieren.
- [x] Neue Daten in einem eigenen Verzeichnis mit vollständigem Manifest,
  Compiler-Version, Modellrevision und Datei-Hashes ablegen.
- [x] Mengenbilanz unabhängig nachzählen und mit dem Manifest abgleichen.
- [x] Konfiguration und aktive Auswahl erst nach erfolgreicher Prüfung auf den
  neuen Korpus binden.

**Abnahme:** Jedes Quellelement ist bilanziert. Alle geeigneten Daten sind im
Training enthalten. Technische Validierung wird nicht als vollständige
inhaltliche Prüfung des gesamten Korpus bezeichnet.

## 5. Vergleichsbasis unter identischen Bedingungen herstellen

- [x] Die bisherigen BiQwen-Gewichte mit bisheriger Kodierung auf vollständigem
  Dev als Ausgangsbasis sichern; bestehende Ergebnisse nur bei nachweislich
  unverändertem Eingabe-, Pipeline- und Präzisionsvertrag wiederverwenden.
- [x] Dieselben Gewichte mit der neuen Typkodierung auf vollständigem Dev
  evaluieren, um den unmittelbaren Eingabeeffekt sichtbar zu machen.
- [x] Die neue Scorer-Initialisierung vor Training prüfen: Ihre beabsichtigte
  Erhaltung der alten Scores numerisch nachweisen.
- [x] Bisheriges veröffentlichtes Tractor-Modell für den Produktvergleich unter
  derselben finalen Bewertungslogik berücksichtigen.
- [x] Reale menschliche Annotationen, deterministische Quellenannotation und
  synthetische Daten getrennt ausweisen.
- [x] Finalwerte, Record-Zuordnung, vollständige Antworten, Mehrdeutigkeiten
  und erwartete Issue-Codes auswerten.
- [x] Unvollständige Referenzen von Auswahlaggregaten ausschließen und fehlende
  reale Evidenz weiterhin ausdrücklich als fehlend markieren.
- [x] Provisorische Schwellen, numerische Präzision und Regression Guards für
  den folgenden Lauf festlegen und in Konfiguration und Berichten binden.
- [x] Klar benennen, welche Baseline die Checkpoint-Auswahl schützt. Vergleiche
  mit alter und neuer Kodierung zusätzlich erhalten, damit ein Eingabeeffekt
  nicht als erlernter Qualitätsgewinn ausgegeben wird.

**Abnahme:** Spätere Verbesserungen lassen sich gegenüber einer nachvollziehbaren
Baseline bewerten; geänderte Eingaben oder Bewertungsregeln bleiben sichtbar.

## 6. Echtes 4B-Modell technisch auf der GPU prüfen

Dieser Schritt ist eine Forward/Backward-Prüfung ohne Optimizer-Updates und ohne
Latenzbenchmark. Er ersetzt kein Training und beweist keine Generalisierung.

- [x] GPU-Belegung und verfügbaren Speicher vor der Prüfung kontrollieren;
  andere Prozesse nicht ungefragt beenden.
- [x] Pinned zusammengeführtes BiQwen-Modell mit neuer Architektur, LoRA,
  vorgesehenen Datentypen und Gradient Checkpointing laden.
- [x] Kurze, lange und recordreiche gültige Requests prüfen, einschließlich
  DE/EN und relevanter Kombinationen nahe den unterstützten Kapazitätsgrenzen.
- [x] Insbesondere den vorgesehenen Microbatch von zwei Requests bei bis zu
  4.096 gemeinsamen Text-/Schema-Tokens auf Speicherverträglichkeit prüfen.
- [x] Forward, Loss, Backward und Gradient Clipping auf endliche Werte prüfen.
- [x] Nichtnull-Gradienten in den vorgesehenen trainierbaren Komponenten
  nachweisen; eingefrorene Komponenten müssen unverändert bleiben.
- [x] Maximale Speicherbelegung und geprüfte Eingabeformen dokumentieren,
  ohne daraus eine Laufzeit- oder Latenzbehauptung abzuleiten.
- [x] Bei Speichermangel Microbatch und Akkumulation nachvollziehbar anpassen,
  möglichst unter Erhaltung der effektiven Batchgröße. Keine Daten oder langen
  Requests stillschweigend auslassen.
- [x] Checkpoint-Roundtrip und Resume-Verhalten mit neuer Architektur durch
  geeignete Softwaretests nachweisen; keinen verdeckten GPU-Pilotlauf anhängen.
- [x] Nachweisen und protokollieren, dass die GPU-Prüfung keine Gewichtsupdates
  durchgeführt hat.

**Abnahme:** Die tatsächliche Hardware trägt den geprüften Trainingspfad. Falls
eine unterstützte Eingabeform nicht trainierbar ist, bleibt dies ein offener
Startblocker statt eines stillen Datenausschlusses.

## 7. Startbedingungen für das Volltraining

- [x] Feldtyp-Encoding und neuer Scorer implementiert und geprüft.
- [x] Gemeinsamer Training-/Inferenzpfad sowie Checkpoint-Kompatibilität geprüft.
- [x] Vollständiger neuer Korpus bilanziert und über Hashes gebunden.
- [x] Baseline und vollständiges Dev unter dem verbindlichen Vertrag verfügbar.
- [x] Diagnostik und Fehlerzählung aktiv.
- [x] Echte 4B-GPU-Prüfung bestanden.
- [x] `nix develop -c just check` erfolgreich ausgeführt.
- [x] Angepassten vollständigen Daten-/Trainings-Preflight erfolgreich ausgeführt.
- [x] Vier vollständige Epochen, Batch-/Akkumulationswerte, Lernraten,
  Loss-Gewichte, Gold-Injektionsplan und Checkpoint-Regeln explizit festgelegt.
- [x] Optimizer-Updates und Präsentationszahl aus der endgültigen Request-Zahl
  neu berechnet; keine alten Zahlen aus dem bisherigen Korpus übernommen.
- [x] Freien Speicher für Daten, Optimizerzustände und Checkpoints geprüft und
  ein neues Ausgabeverzeichnis festgelegt.
- [x] Aktive Konfiguration eindeutig auf genau diesen Lauf gesetzt; Status
  weiterhin `prepared`, solange das Training tatsächlich noch nicht läuft.

**Abnahme:** Der Lauf ist technisch und fachlich startbereit. Das Abarbeiten
dieser Liste allein startet ihn nicht; die Ausführung folgt dem Trainingsauftrag.

## 8. Vier vollständige Epochen trainieren

- [x] Den vollständigen Lauf über den verbindlichen Nix-/Tractor-Einstieg starten,
  nicht über den gebündelten Notebook-Trainer.
- [x] Status erst beim tatsächlichen Start auf laufend setzen.
- [ ] Pro Epoche jeden akzeptierten Request genau einmal verarbeiten, einschließlich
  des letzten unvollständigen Batches.
- [ ] Keine `max_steps`-Abkürzung, Stichprobe, Domänenquote oder vorzeitige
  Beendigung als abgeschlossenes Volltraining ausgeben.
- [ ] Bisherige Datenverteilung zunächst erhalten. Zusätzliche Loss-Gewichtungen
  nur ausdrücklich konfigurieren, begründen und anhand vollständigen Dev bewerten.
- [ ] Checkpoints regelmäßig für die Wiederaufnahme speichern; Daten-, Code-,
  Architektur- und Konfigurationsbindung bei Resume prüfen.
- [ ] Nicht endliche Losses, Speicherfehler oder fehlerhafte Beispiele nicht
  überspringen. Fehlerzustand dokumentieren und nach Korrektur sauber fortsetzen.
- [ ] Nach jeder Epoche den vollständigen Abdeckungsnachweis erzeugen.
- [ ] Nach jeder vollständigen Epoche alle Dev-Dokumente durch dieselbe
  Produktionspipeline auswerten, ohne Gold-Injektion oder Antwort-Hilfen.
- [ ] Nur vollständige Epochen als Modellkandidaten auswählen; Regression Guards
  und Auswahlmetrik auf den finalen Pipeline-Ergebnissen anwenden.
- [ ] Lernkurven, Kandidatenabdeckung und Ergebnisse nach DE/EN, Domäne und
  Ausgabeform nachvollziehbar berichten.
- [ ] Abschluss erst nach allen vier vollständigen Epochen markieren.

**Abnahme:** Abdeckung und tatsächliche Trainingsschritte sind belegbar. Ein
vollständig trainierter, aber qualitativ unzureichender Lauf bleibt als solcher
ausgewiesen und wird nicht automatisch zum Release.

## 9. Qualität, Kalibrierung und verbleibende Grenzen bewerten

- [ ] Den besten zulässigen Volltrainings-Checkpoint anhand vollständigen Dev
  auswählen und Auswahlentscheidung dokumentieren.
- [ ] Finales Feld-/Record-Ergebnis sowie Issue- und Zurückweisungsqualität
  gegenüber beiden BiQwen-Eingabevarianten und Tractor vergleichen.
- [ ] Das Ziel von über 70 % je vereinbarter Domäne und Ausgabeform prüfen;
  DE/EN separat ausweisen und geringe Fallzahlen sichtbar machen.
- [ ] Produktqualität als sekundären Regression Guard behandeln; sie darf
  Schwächen allgemeiner Aufgaben nicht verdecken.
- [ ] Fehlende reale Evaluationsdaten nicht durch hohe synthetische Werte ersetzen.
- [ ] Nach festgelegter Modellauswahl Schwellen auf dem separaten
  Kalibrierungssplit bestimmen; Kalibrierungsdaten nicht zur Modellwahl verwenden.
- [ ] Die festgelegte Pipeline abschließend auf dem unberührten Testsplit prüfen.
- [ ] Verbleibende Fehler nach Vorschlag, Auswahl, Zuordnung, Normalisierung,
  Kapazität und Referenzlücke zusammenfassen.
- [ ] Nur bei passender Evidenz eine Vergrößerung des bidirektionalen Adapters
  oder einen Backbone-Umbau planen. Nicht pauschal aus niedrigem Gesamt-F1 auf
  unzureichende Adapterkapazität schließen.
- [ ] Bei einer weiteren Architektur- oder Rezeptänderung diese versionieren
  und neue Qualitätsbehauptungen erneut belegen; Testdaten nicht zur Abstimmung
  des nächsten Modells verwenden.
- [ ] API-Smoke-Test mit dem ausgewählten Checkpoint einschließlich Laden,
  Extraktion, Normalisierung und Issues durchführen.
- [ ] Endstatus getrennt dokumentieren: Daten vollständig, Training vollständig,
  technische Prüfungen bestanden, Qualitätsziele erreicht oder verfehlt.

**Abnahme:** Es ist belegt, welche Fähigkeiten die vollständige API verbessert
hat und welche Ziele offen bleiben. Upload, Push und Veröffentlichung sind keine
automatischen Schritte dieses Plans.

## Reihenfolge und Ergebnisnachweise

`Ausgangszustand → Typkodierung → Record-Scorer → Diagnostik → vollständiger
Datenaufbau → Baselines → GPU-Prüfung → Startbedingungen → Volltraining →
Kalibrierung und Abschlussbewertung`

Für jeden abgeschlossenen Abschnitt sind im Arbeitsbericht mindestens Commit,
relevante Konfiguration/Hashes, ausgeführte Prüfung, Ergebnis und verbleibende
Einschränkungen zu vermerken. Statusberichte müssen vorbereitete Arbeit und
tatsächlich ausgeführte Rechenschritte eindeutig unterscheiden.

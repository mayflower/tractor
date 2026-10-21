# tractor: Anforderungen und Ausbauplan für strukturierte Extraktion

CORD ist vollständig und dauerhaft ausgeschlossen: keine Trainings-, Dev-,
Kalibrierungs- oder Testeinträge, auch keine Kopffelder oder abgeleiteten Fenster.

Stand: 19. Oktober 2026. Verbindliche fachliche Zieldefinition für den nächsten
Daten- und Trainingsstand. Dieses Dokument plant die Umsetzung; es behauptet
nicht, dass neue Quellen bereits importiert oder bestehende Gewichte bereinigt sind.
Bei fachlichen Widersprüchen ersetzt diese Abgrenzung ältere Generalisten-Ziele.
Die technischen Konfigurationen und AGENTS.md müssen in Phase 0 daran angeglichen werden.

## 1. Produktziel und Grenzen

tractor übernimmt **ausdrücklich im Eingabetext stehende Werte in ein vorgegebenes
Zielschema**. Unterstützte Ergebnisformen sind:

1. Einzelobjekte mit benannten Feldern, etwa Rechnungskopf oder Vertragsstammdaten.
2. Objekte mit wiederholten Datensätzen, deren Werte und Zusammengehörigkeit
   ausdrücklich im verständlichen Text beschrieben sind, etwa Beschäftigungsstationen.
3. Typisierte Entitätenlisten, etwa Personen, Organisationen, Orte und Identifikatoren.

Jeder ausgegebene Wert muss intern auf eine konkrete Stelle des Eingabetexts zurückführbar
sein. Originalschreibweise und Zeichenpositionen bleiben für interne Validierung nachvollziehbar.
Schema und Feldbeschreibungen bestimmen, welche Werte verlangt werden.
Nicht erwähnte Werte werden nicht ergänzt. Fehlende Annotation ist kein Beleg
für einen tatsächlich fehlenden Wert.

### Explizit außerhalb des Produkts

- Jede Klassifikation, jedes Tagging und Routing: auch Freigabestatus-, Dokumentart-
  und Anliegenart-Klassifikation; ferner Dringlichkeit, Sentiment und Assistant-Intent.
- Zusammenfassungen, freie oder extraktive QA, Suche und Klausellokalisierung.
- Relationsextraktion als eigenständiges Ziel, insbesondere Head/Tail-Tripel und Graphkanten.
- Matching, Entity Linking und Normalisierung auf ICD-, SKU-, Skill- oder andere Codesysteme.
- OCR, Bilderkennung und Transkription als Modellaufgabe.
- Tabellenparsing, Spalten-/Zeilenrekonstruktion und layoutabhängige Positionszuordnung,
  auch aus bereits transkribierten PDF-/OCR-Texten. Keine entsprechende Trainings- oder Freigabeanforderung.
- Coding-Agent-Hilfsextraktion, beispielsweise Dateipfade aus Kommandoausgaben.
- Berechnungen, Weltwissen, erschlossene Ursachen, nicht genannte Tatsachen und explizites Reasoning.

Ein im Text vorhandener Code darf unverändert extrahiert werden. Ein Code darf
nicht aus einer Beschreibung hergeleitet werden. Ein gedruckter Gesamtbetrag darf
extrahiert, aber nicht aus Einzelbeträgen errechnet werden. Ein wörtlicher Status
kann als Textfeld vorkommen; ihn in eine vorgegebene Klasse zu übersetzen ist ausgeschlossen.

Record-Extraktion übernimmt ausdrücklich textuell beschriebene Datensätze.
Die Zusammengehörigkeit von Tabellenspalten oder räumlich angeordneten Belegpositionen
zu rekonstruieren ist ausgeschlossen. Verständliche Aussagen wie „Anna begann am
1. März bei Beispiel GmbH“ bleiben zulässige Record-Extraktion.
Ereignisbeispiele sind nur zulässig, wenn sie ausdrücklich genannte Werte in ein
konkretes Geschäftsschema übernehmen, etwa Zeitpunkt, Anlage und Meldetext eines
Incident-Berichts. Ein technischer Familienname allein entscheidet nicht über Zulässigkeit.
Abschnittsübergreifende Zuordnung ist nur bei expliziter Identität/Referenz zulässig.
Keine verdeckte Wiederaufnahme von Relations- oder QA-Aufgaben durch Umbenennung.

### Ein- und Ausgabe

- Eingabe: Verständlicher Text mit ausdrücklich benannten Werten und ein Zielschema.
- Vorhandene OCR-Transkripte eines öffentlichen Datasets können Texteingaben sein.
  tractor erhält keine Bildaufgabe; der Adapter darf keine Gold-Labels in den Text einbauen.
- Ziel: gemäß Feldtyp normalisierte Werte; öffentliche Antwort ausschließlich `data` und `issues`.
  Keine Evidenzoption und keine freie Antwortgenerierung.
- Kopffelder und ausdrücklich im Text beschriebene Datensätze können gemeinsam abgebildet werden. Flache Record-Listen
  sind zunächst ausreichend; beliebig rekursives JSON ist keine behauptete Fähigkeit.
- Deutsch und Englisch sind die primären Sprachen. Andere Sprachen separat ausweisen;
  Übersetzungen sind abgeleitete Daten und bleiben mit dem Original im selben Split.

## 2. Ausgangslage und verbleibende Bereinigung

Bestandsaufnahme aus `data/generalist-extraction-scoped/manifest.json`,
`src/tractor/synthetic_coverage.py` und einer Zählung der Trainingsdatei.
Die folgenden Zahlen beschreiben den Stand vor Umsetzung dieses Plans.

| Bestand | Beispiele | Einordnung |
|---|---:|---|
| Training insgesamt | 233.537 | Bereits ohne freie QA, MASSIVE, Routing, Dringlichkeit und Sentiment |
| Few-NERD und DocRED | 125.985 | Öffentliche, menschlich annotierte Entitätenextraktion; derzeit 54 % des Trainings |
| Relationsextraktion | 20.001 | Noch zu entfernen |
| Verbleibende Klassifikation | 6.000 | Noch zu entfernen, einschließlich gemischter Episoden |
| Synthetische Record-Familie | 25.001 | Grundlage, aber überwiegend sehr kurze und einfache Vorlagen |
| Ereignisfamilie | 15.001 | Fachlich einzeln prüfen und gegebenenfalls auf zulässige Record-Schemata begrenzen |
| Abschnittsverknüpfungen | 10.001 | Explizite Zuordnung zu Records prüfen; keine allgemeine Relationsaufgabe |
| Dokumentfelder | 16.382 | Davon 15.000 synthetische Kontakt-/Referenz-/E-Mail-Aufgaben |
| Text-to-JSON-Import insgesamt | 1.371 | Viele Themen, aber geringe Tiefe; Herkunft/Annotation nicht als human verifiziert |
| Importierte Lebensläufe | 174 | Reale Texte mit Teacher-Labels; bisher im Wesentlichen Jobtitel |

Nach Entfernung der 20.001 Relations- und 6.000 Klassifikationsbeispiele bleiben
rechnerisch **207.536** Trainingsbeispiele vor weiterer fachlicher Prüfung.
Diese Zahl ist weder ein neues Qualitätsziel noch ein Nachweis ausreichender Abdeckung.

Die Texte sind kurz: Median 24 Wörter, 90. Perzentil 41, 99. Perzentil 130 Wörter.
Die synthetischen Hauptaufgaben verwenden ein bis fünf Felder und meist zwei bis
vier Records. Viele Wiederholungen stammen aus wenigen Render- und Wertvorlagen.

Dev enthält 5.995 Beispiele, darunter nur 229 real/human aus DocRED und 5.600
deterministisch synthetische Beispiele. Few-NERD fehlt im aktuellen Dev-Split.
Kalibrierung und Test enthalten nach der bisherigen Bereinigung keine real/human
Beispiele. Unbekannte Herkunft und Teacher-Labels werden gesondert gezählt.
Das bestehende Evaluationsset belegt daher keine breite Geschäftsdokument-Zuverlässigkeit.

## 3. Fachliche Lücken und Priorität

Die vom Auftraggeber vorgelegte 16-Domänen-Liste dient als Markt- und
Priorisierungsrahmen. Ihre AEI-Anteile sind Schätzungen, keine unabhängig geprüften
Trainingsquoten. Die Priorisierung der vier großen Einsatzbereiche ist wichtiger
als eine scheinpräzise Nachbildung der Prozentwerte.

| Prio | Domäne | Fehlende Abdeckung | Geplanter Aufbau |
|---|---|---|---|
| P1 | Rechnungen, Belege, Bestellungen | Vollständiger Kopf plus Positionen; Rechnungs-/Bestell-/Kundennummer; Netto, Steuer, Brutto, Rabatt; Bankdaten und Betragsrollen | Öffentliche Beleg-/Positionsdaten zuerst; synthetische vollständige Rechnungen und Bestellungen ergänzen |
| P1 | Verträge, Recht, Immobilien | Parteien und Rollen, Anschriften, Beginn/Ende, wörtliche Kündigungsfrist, Vergütung, Objekt-/Aktenzeichen | Eng begrenzte Stammdatenadapter; synthetische Miet-, Dienstleistungs- und Arbeitsverträge ohne Klauselsuche |
| P1 | Kommunikation → Vorgangsfelder | Gleichzeitige Kunden-, Vertrags-, Bestell- und Schadennummern; Ansprechpartner, Rückrufnummer, Termine, Mengen, Artikel | Nachrichten und bereits transkribierte Gespräche mit expliziten Feldern; keine Intent- oder Prioritätslabels |
| P1 | KYC, Bank, Kredit | Identitätsfelder, Ausweisnummer, Arbeitgeber, explizites Einkommen, Kontoverbindung; wiederholte Umsätze | Synthetische Identitäts-/Kreditunterlagen und Kontoauszüge; reale Quellen separat erschließen |
| P2 | Anträge, Formulare, Behörden | Zusammenhängende Antragstellerdaten, Bezugspersonen, Adressen, Kennungen, Beträge und Daten | Öffentliche Formular-KIE-Daten plus deutsche/englische synthetische Anträge |
| P2 | Versicherung | Policenkopf, Schadenmeldung, Summen, Selbstbehalt, Beteiligte, Schadenhistorie und Positionen | Synthetische fachliche Schemata; vorhandene kleine Importfelder nur als Ergänzung |
| P2 | Klinische Dokumente | Medikamentenlisten mit Dosis/Frequenz, Labor-/Vitalwerte mit Einheit und Datum, explizite Befundangaben | Synthetische Textberichte aus kontrollierten Fakten; keine Diagnoseableitung oder Codezuordnung |
| P2 | Lebensläufe, Stellenanzeigen | Vollständige Personalien, Stationen, Zeiträume, Abschlüsse, wörtlich genannte Skills, Gehalt, Standort | Bestehenden CV-Import auf zulässige Felder erweitern; unterschiedliche CV-/Stellenformate synthetisch ergänzen |
| P2 | Firmen- und Kontaktdaten | Firma, Adresse, Funktion, Telefon, E-Mail, explizite Mitarbeiterzahl und Branchenangabe | Sichtbare Webseiten-/Signaturtexte mit belegten Zielwerten; keine erschlossene Branche |
| P2 | Produkt- und Lieferantendaten | Marke, Material, Farbe, Größe, Maße, Gewicht, Inhaltsstoffe und wörtliche Zertifikate | Attributextraktionsdaten ohne Matching/Normalisierung; heterogene Produkttexte ergänzen |
| P2 | Dokumentmetadaten | Absender/Empfänger, Datum, Nummer, Aktenzeichen und Betreff als gemeinsames Objekt | Querschnitt aller Dokumentgeneratoren; keine Dokumenttyp-Klassifikation |
| P3 | Finanz-/Unternehmensberichte | Ausdrücklich genannte Kennzahl, Originalwert, Einheit und Periode | Textgebundene Annotation aus Berichten; XBRL nur als getrennte Annotationhilfe |
| P3 | Wissenschaftliche Literatur | Messwert, Einheit, Bedingungen, Stichprobe, explizites Studiendesign, Bibliografie | Kleine überprüfte öffentliche Teilmengen und synthetische Studien-/Messwertauszüge |
| P3 | News und Social Media | Vorgegebene Objekte aus Firma, Person, Ort, Datum, Betrag und Ticker | Bestehende NER-Daten nur ergänzend; keine Ereignisklassifikation oder Relationslabels |
| P3 | Logistik und Zoll | Absender/Empfänger, Sendung, Container, Häfen, Gewichte, gedruckte Zollnummer, Warenpositionen | Synthetische Frachtbriefe/Packlisten plus geeignete öffentliche Formulare |
| P3 | Operative/technische Berichte | Anlage, Komponente, Zeitpunkt, Mess-/Grenzwerte und ausdrücklich genannte Maßnahmen | Vorhandene Incident-/Log-Beispiele fachlich begrenzen; Wartungs- und Prüfberichte ergänzen |

P3 bedeutet spätere Umsetzung, nicht einen niedrigeren Qualitätsanspruch für eine
anschließend beworbene Domäne.

## 4. Öffentliche Quellen: Kandidaten und Aufnahmebedingungen

Recherche am 17.10.2026 anhand der unten verlinkten Primärquellen.
**Kandidat bedeutet nicht importiert oder für Veröffentlichung freigegeben.**
Vor jedem Import Revision, Dateien, Hashes, Datenlizenz, Herkunft, Annotationsart,
verfügbare Splits und textuelle Belegbarkeit erfassen. Eine Code-Lizenz allein
ist keine Freigabe der Dokumente. Nutzungsbeschränkte Quellen dürfen keine
Grundlage der geplanten offenen Veröffentlichung werden.

| Quelle | Sinnvoller Beitrag | Adapter und Grenzen | Reihenfolge |
|---|---|---|---|
| [DocILE](https://docile.rossum.ai/) / [Repository](https://github.com/rossumai/docile) | Geschäftsdokument-Kopffelder ohne Positionsparsing; Anbieter nennt 6.680 annotierte und 100.000 synthetische Dokumente | Vorhandenen Text/OCR deterministisch serialisieren, nur ausdrücklich lesbare Einzelfelder übernehmen; visuelle Lokalisierungslabels nicht trainieren. Textverlust/Mehrdeutigkeit prüfen. Datenbedingungen vor Aufnahme klären. Synthetischer Teil zählt nicht als reale Quelle; unannotierter Teil nicht als fertige Trainingsdaten. | P1, zuerst Machbarkeit und Bedingungen |
| [VRDU](https://github.com/google-research-datasets/vrdu) | Explizite Stammdaten aus Registrierungsformularen und Anzeigenaufträgen | Vorhandene OCR-Texte und strukturierte Annotationen nutzen; offizielle Splits erhalten. Keine Layout-Generalisation oder Positionszuordnung trainieren; nur explizite Einzelfelder übernehmen; Datenbedingungen gesondert prüfen. | P1/P2 |
| [Kleister NDA](https://github.com/applicaai/kleister-nda) | Vertragsstammdaten | Nur wörtlich belegbare Parteien-/Datums-/Fristwerte. Falls Zielwerte normalisiert sind, belegte Originalspannen annotieren oder Beispiel ausschließen. Kein Normalisierungsbenchmark und keine Klausellokalisierung. | P1 |
| [Kleister Charity](https://github.com/applicaai/kleister-charity) | Organisations- und Finanzberichtsfelder in längeren Dokumenten | Originalwerte statt kanonisierter Zielstrings; Dokument-/Periodenbindung prüfen. Textfenster nur ohne Gold-Leakage bilden. | P2/P3 |
| [suneeldk/text-json](https://huggingface.co/datasets/suneeldk/text-json) | Kleine Ergänzung vieler Feldschemata; bereits lokal vorhanden | Jeden Fall erneut gegen den engen Scope prüfen. Keine Aufgaben mit berechneten, klassifizierten, normalisierten oder Coding-Agent-Zielen. Herkunft weiter unknown, solange nicht belegt. | P1, Bestand nachprüfen |
| [resume-json-extraction-5k](https://huggingface.co/datasets/sandeeppanem/resume-json-extraction-5k) | Kandidatenprofile und wiederholte Stationen; bereits teilweise importiert | Nur belegte Felder; keine Skill-Taxonomie, Eignungsbewertung oder rekonstruierten Angaben. Verlust durch Kontext-/Schema-Grenzen messen. Teacher-Labels gesondert führen. | P2 |
| [WDC-PAVE](https://webdatacommons.org/structureddata/wdc-pave/) | Produktattribute; 1.420 Angebote mit 24.582 annotierten Attribut-Wert-Paaren laut Quelle | Nur Originalwerte aus Titel/Beschreibung. Normalisierten Goldstandard und Produktmatching ausschließen. Attributpaare nicht als unabhängige Dokumente zählen. | P2 |
| [SEC EDGAR](https://www.sec.gov/search-filings) | Öffentliche Finanzberichte als Quellenpool | Noch kein fertiger Extraktionskorpus. Sichtbaren Text mit belegten Originalwerten annotieren; XBRL darf Hinweise liefern, aber keine Taxonomie-/Skalierungsaufgabe werden. Eingabe von Annotationstags bereinigen; Vorzeichen, Einheit und Periode verifizieren. Nutzungsbedingungen/Zugriffsvorgaben beachten. | P3, separate Silver-Pipeline |
| [Synthea](https://github.com/synthetichealth/synthea) | Strukturierte synthetische Patientenfakten als Ausgangspunkt | Daraus unterschiedliche Textberichte rendern und nur gedruckte Werte extrahieren. Keine echten klinischen Daten, kein fertiger Text-Extraktionsbenchmark und keine Code-Normalisierung. | P2, synthetische Quelle |

Für KYC/Kredit, Versicherung, Geschäftskommunikation und Logistik ist durch diese
Recherche noch **kein ausreichend breiter, unmittelbar passender öffentlicher
Gold-Korpus belegt**. Diese Lücke wird nicht durch beliebige thematisch benannte
HF-Datasets oder die bloße Existenz öffentlicher Dokumente als geschlossen gewertet.
Quellenbeschaffung und synthetischer Aufbau laufen dort getrennt.

Nicht importieren: SQuAD, GermanQuAD, AdversarialQA, MultiSpanQA, MASSIVE,
Function-Calling-/Tool-Use-Korpora, Text-to-SQL, reine Schema-Benchmarks und
Relationsbenchmarks. [CUAD](https://github.com/The-Atticus-Project/cuad) ist ein
Klauselreview-Dataset und wird nicht als Ganzes aufgenommen.
[FinanceBench](https://arxiv.org/abs/2311.11944) ist Finanz-QA und gehört nicht in
unsere Extraktionsmischung. Thematische Nähe allein ist kein Aufnahmekriterium.

## 5. Mengenplanung und ausgewogene Mischung

### Reale öffentliche Daten

Die bisherige Vorgabe **mindestens 200.000 Beispiele aus echten öffentlichen
Extraktionsquellen** bleibt bestehen. Aktuell sind konservativ 125.985 real/human
belegt; es fehlen mindestens **74.015** vor eventuellen weiteren Ausschlüssen.
Die oben genannten Kandidaten garantieren diese Menge ausdrücklich nicht.

Separat zählen: eindeutige Quelldokumente, Dokument-Schema-Paare, Textfenster,
Paraphrasen, Trainingswiederholungen und unabhängige Quell-/Vorlagenfamilien.
Eine Aufteilung in Felder oder überlappende Fenster erhöht nicht die Zahl
unabhängiger Quellen. Unannotierter Text ist kein fertiges Extraktionsbeispiel.
Reale Texte mit neuen Teacher-Labels bleiben real/teacher; die Anzahl geprüfter
human-Labels wird nicht dadurch erhöht. Synthetische Dokumente zählen nie zum Realziel.

Das Realziel nicht mit noch mehr gleichartiger NER auffüllen. Nach dem Quellenpilot
ist ein Mengenbericht mit belastbarem Ertrag je Quelle und verbleibender Lücke
Pflicht. Falls 200.000 nicht seriös erreichbar sind, bleibt die Anforderung offen;
kein Umdeklarieren synthetischer Daten und kein automatischer Ersatz durch QA.

### Synthetischer Ausbau

Planungsziel: **200.000 neue, validierte Trainingsszenarien**, zusätzlich getrennte
Evaluationsszenarien. Das ist ein Ausbauziel, kein Startauftrag für einen unkontrollierten
Großlauf. Erst 30.000 prüfen, dann 100.000, dann 200.000. Bestehende passende Beispiele
können angerechnet werden, wenn sie dieselben Qualitäts-/Diversitätsprüfungen bestehen;
beides im Mengenbericht getrennt ausweisen.

| Domäne | Plananteil | Ziel bei 200.000 |
|---|---:|---:|
| Rechnungen/Belege/Bestellungen | 18 % | 36.000 |
| Verträge/Recht/Immobilien | 12 % | 24.000 |
| Anträge/Formulare/Behörden | 5 % | 10.000 |
| KYC/Bank/Kredit | 10 % | 20.000 |
| Kommunikation/Vorgangsfelder | 12 % | 24.000 |
| Firmen-/Kontaktdaten | 6 % | 12.000 |
| Versicherung | 7 % | 14.000 |
| Klinik | 7 % | 14.000 |
| Lebensläufe/Stellenanzeigen | 6 % | 12.000 |
| Produktattribute | 5 % | 10.000 |
| Dokumentmetadaten | 3 % | 6.000 |
| Finanzberichte | 2 % | 4.000 |
| Wissenschaft | 2 % | 4.000 |
| News/Social Media | 1 % | 2.000 |
| Logistik/Zoll | 2 % | 4.000 |
| Operative/technische Berichte | 2 % | 4.000 |
| **Summe** | **100 %** | **200.000** |

Dies sind technische Startquoten, keine behaupteten Marktanteile. Pro Beispiel
eine primäre Domäne für die Mengenzählung, zusätzliche Coverage-Tags erlaubt.
Metadaten als Nebenfelder anderer Domänen nicht doppelt als Beispiel zählen.

Über die gesamte tatsächliche Trainingsmischung zunächst **60 % Einzelobjekte,
30 % Objekte mit wiederholten Records, 10 % Entitätenlisten** als obere Zielquoten anstreben.
Fehlende Vielfalt darf nicht durch erzwungene Wiederholung verdeckt werden:
synthetische Beispiele höchstens dreimal pro Epoche verwenden und nicht gefüllte
Quoten ausweisen. Reine Organisationsmetadaten aus Finanzberichten zählen nicht
als Nachweis von Kennzahlenextraktion.
Ein Kopf-plus-Positionen-Beispiel zählt primär zur zweiten Form. Die vielen
vorhandenen NER-Beispiele sind ein Pool, keine Verpflichtung, sie in jeder Epoche
alle zu verwenden. Effektive Expositionen, Unique Counts und Quoten getrennt berichten.
DE/EN zunächst 50/50 innerhalb des neuen synthetischen Anteils; öffentliche
Sprachverteilungen bleiben nachvollziehbar und werden nicht umetikettiert.

## 6. Anforderungen an synthetische Daten und Adapter

### Fakten zuerst, Text danach

1. Ein typisiertes Szenario mit festen Originalwerten und Record-Identitäten erzeugen.
2. Vor der Textgenerierung Split, Domäne, Dokument-/Vorlagenfamilie und Sprache festlegen.
3. Unterschiedliche Texte aus diesen Fakten rendern: Fließtext, E-Mail, Formular,
   natürlich formulierte Absätze und ausdrücklich beschriebene Einzelvorgänge.
   Keine Tabellen oder Positionsblöcke als Record-Aufgabe erzeugen.
4. Zielwerte und Evidenzspannen aus der tatsächlich gerenderten Eingabe ableiten.
   Bei LLM-Paraphrasen Spannen neu verifizieren; gleiche Werte an mehreren Stellen
   dürfen nicht blind per erster Zeichenkettenfundstelle zugeordnet werden.
5. Jeden Wert auf Textbeleg, richtige Feldrolle, Record-Zugehörigkeit und Schema prüfen.
   LLM-Konsens allein ersetzt diesen Nachweis nicht.
6. Nicht belegbare oder nur erschlossene Werte verwerfen; keine Reparatur durch Halluzination.

LLMs dienen primär der sprachlichen und strukturellen Vielfalt. Gold-Werte stammen
nicht aus einer unkontrollierten zweiten Extraktion desselben Teachers.
Erzeugung und unabhängige Qualitätsprüfung sind getrennte Schritte.

### Erforderliche Variation

- Pro P1-Domäne mindestens 20 unterscheidbare Render-/Vorlagenfamilien und mehrere
  Schemaformulierungen; neue Namen/Zahlen allein gelten nicht als neue Vorlage.
- Kurze und längere Feldschemata: 3–8, 9–20 und 21–40 Felder. Große Schemata nur
  innerhalb nachgewiesener Modellkapazität oder ausdrücklich als Pipeline-Aufgabe.
- Ein oder mehrere ausdrücklich beschriebene Records innerhalb der Modellkapazität;
  leere Listen nur nach geprüftem nativen Trainingsvertrag. Keine Tabellenlängen- oder Layoutanforderung.
- Wiederkehrende Namen, mehrere Datums-/Betrags-/Kennungsrollen, unterschiedliche
  Währungen und Schreibweisen; Originaltext extrahieren, nicht normalisieren.
- Fehlende optionale Angaben, belegte Null-/Nullbeträge, mehrzeilige Werte,
  irrelevant ähnliche Werte und ausdrücklich ersetzte Angaben.
- Beträge nicht nur als ganze EUR-Zahlen: Dezimalstellen, Tausendertrennzeichen,
  negative gedruckte Werte, Rabatte und getrennte Netto-/Steuer-/Bruttobeträge.
- Rechnungsnummer, Kundennummer, Bestellnummer und Kontokennung im selben Dokument.
- Anschriften, Freitextbeschreibungen, mehrteilige Namen und unveränderte Einheiten.
- Dokumentartefakte wie Kopf-/Fußzeilen und Abschnittsüberschriften;
  kein künstlicher Gold-Hinweis im Eingabetext.

Für künstliche Personal-, Bank- und Gesundheitsangaben ausschließlich fiktive
Identitäten verwenden. Synthetische Herkunft bleibt in jeder Ableitung erhalten.

### Kontextgrenzen offen behandeln

Der bisher geprüfte native Verarbeitungspfad hat ein gemeinsames Kontextbudget
von etwa 512 Tokens für Text und Schema. Vor großen Schemas oder längeren Texten
Kapazität mit dem tatsächlich verwendeten Modell/Tokenizer messen. Ein größeres
Konfigurationsfeld allein beweist keine größere Encoder-Kapazität.

Kein stilles Abschneiden von Gold-Werten. Der Adapter meldet verworfene Dokumente,
verlorene Felder und nicht rekonstruierbare Records. Falls nötig, dokumenttreue,
gold-unabhängige Fensterbildung und deterministische Zusammenführung implementieren.
Fenster dürfen nicht anhand der Gold-Antwortposition ausgewählt werden.
Record-Zuordnung über Fenster nur mit ausdrücklichem Anker, sonst als nicht
unterstützt melden. Native Extraktion und Gesamtpipeline getrennt evaluieren.
Die Zielabdeckung vollständiger Dokumente bleibt sichtbar, auch wenn ein erster
Modellstand nur Ausschnitte unterstützt.

## 7. Evaluation und Freigabekriterien

Verbindlich ist die vollständige Produktionspipeline: Extraktion, deterministisches
Parsing, Validierung, programmatische Normalisierung und Mehrdeutigkeitsbehandlung.
Checkpoint-Auswahl, LLM-Vergleiche und Freigaben bewerten die endgültigen API-/SDK-
Feldwerte und Records. Native Spannenmetriken dienen ausschließlich der Diagnose.
Originalspannen bleiben Belege; wiederholte Nennungen desselben Werts sind bei
Unique-Listen keine zusätzlichen Geschäftswerte. Verschiedene Records bleiben erhalten.
Evaluation und API verwenden denselben Verarbeitungscode. Erwartete Endwerte werden
unabhängig definiert; Gold darf keine Parser- oder Kandidatenentscheidungen steuern.
Normalisierung ausdrücklich belegter Werte ist Produktumfang; fehlende Fakten,
Tabellenrekonstruktion und Taxonomie-Zuordnung bleiben ausgeschlossen.


### Neue Evaluation nach Fachdomäne und Ausgabeform

Die bisherigen technischen Familien ersetzen keine Domänenprüfung. Einen neuen
Coverage-Bericht für alle 16 Domänen, drei Ausgabeformen, Sprachen, Werttypen,
Schemagrößen und Textlängen anlegen. Zusätzlich real/human, real/teacher,
unknown und synthetic getrennt ausweisen.

- Für jede Domäne mindestens 200 unabhängige Szenarien je Dev-, Kalibrierungs-
  und Testsplit planen: zunächst 3.200 Fälle je Split, außerhalb des Trainingsziels.
- Für jede P1-Domäne mindestens 100 echte, überprüft annotierte Dokumente je
  Evaluationssplit anstreben. Das sind Mindestplanwerte, kein Ersatz für ein
  Unsicherheitsintervall; Verfügbarkeit nach Quellenpilot belegen.
- Keine Real-Fälle verfügbar: synthetische Qualität berichten, Real-Abdeckung als
  offen markieren. Keine Freigabeaussage für reale Dokumente dieser Domäne.
- Offizielle Testdaten niemals ins Training übernehmen. Bereits verwendetes Dev
  nicht nachträglich zu einem unabhängigen finalen Test erklären.
- Alle Ausschnitte, Übersetzungen, Paraphrasen und Varianten eines Dokuments/
  Szenarios im selben Split halten. Wiederverwendete Vorlagen, Organisationen
  und Dokumentfamilien über geeignete Holdouts testen; Near-Duplikate prüfen.
- Heldout-Texte nicht allein aus denselben Templates mit anderen Zahlen erzeugen.

### Metriken

- **Mehr als 70 % Field-F1 in jeder erklärten Fachdomäne**, separat für die
  vorhandenen realen und synthetischen Evaluationsgruppen.
- **Mehr als 70 % in jeder Ausgabeform**; für wiederholte Datensätze zusätzlich
  Record-F1 mit eins-zu-eins-Zuordnung, damit vertauschte Positionen auffallen.
- Identifikatoren, Beträge/Mengen, Datums-/Fristwerte, Namen, Kontakt-/Adresswerte
  und beschreibende Attribute jeweils getrennt berichten.
- Vollständige Objekt-/Request-Exact-Rate, Präzision, Recall, fehlende Werte,
  unerlaubte Zusatzwerte und Abstention bei tatsächlich fehlender Angabe ausweisen.
- Exakte Textwerte und Evidenzpositionen prüfen. Alternative zulässige Belegstellen
  explizit annotieren; Positionsfehler und inhaltlich falsche Werte separat diagnostizieren.
- Modell-/Pipelinefehler oder fehlende Messungen führen nicht zum Herausnehmen
  des Beispiels aus dem Nenner.
- Konfidenzintervalle nach unabhängigen Dokument-/Szenariogruppen berichten;
  Wiederholungen und Fenster erhöhen nicht künstlich die Stichprobengröße.

Makro-Durchschnitt und Produktregression dürfen einen schwachen Bereich nicht
kompensieren. Die bestehende Produktregressionsgrenze von mindestens 90 % bleibt
sekundär; sie ist weder Mischungsquote noch Begründung für erneute Spezialisierung.
Nur Dev wählt Modell und Rezept. Erst danach Schwellen auf separater Kalibrierung
festlegen und den eingefrorenen Test auswerten. Keine GPU-Latenzprüfung im Ausbauplan.

## 8. Ausführbarer Arbeitsplan mit Prüfpunkten

| Phase | Arbeit | Konkretes Ergebnis / Abschlussbedingung |
|---|---|---|
| 0 — Scope vereinheitlichen | Verbliebene Relations-/Klassifikationsdaten aus allen aktiven Splits entfernen; Events, Abschnittsverknüpfungen und Text-to-JSON-Fälle fachlich prüfen; Instruktionen, Generatoren, Mischung und Gates angleichen | Versioniertes Scope-Inventar, Ausschlussgrund je Familie/Fall, Hashes, Tests gegen Wiederaufnahme. Alte Ergebnisse nicht rückwirkend als Erfolg unter neuem Scope darstellen. |
| 1 — Schemata und Quellenpilot | 16 Domänen definieren; zunächst P1-Schemata vollständig ausarbeiten. Je priorisierter öffentlicher Quelle bis zu 200 Dokumente prüfen, bei kleineren Quellen verfügbare Menge | `schemas/` mit Feldbeschreibungen und Record-Ankern; Quellenregister mit Revision, Lizenz, Herkunft, Split und realistischem Importertrag. Mindestens 50 manuell kontrollierte Fälle je Adapter, soweit verfügbar. |
| 2 — Adapter und neue Evaluation | DocILE/VRDU/Kleister nach Eignungsprüfung anbinden; Bestandsimporte neu prüfen. Originalwerte, Record-Gruppen und Textserialisierung testen. Gold-unabhängige Kontextstrategie entscheiden | Gepaarte Text-/Native-Dateien, Evidenz- und Reject-Bericht, geschützte Evaluationssplits. Keine Quelle bei stiller Gold-Verlustrate als vollständig importiert melden. |
| 3 — Synthetischer Pilot 30.000 | Faktenmodelle und vielfältige Renderer aufbauen; Quoten proportional zur 16-Domänen-Tabelle. Nach Herkunft/Template gruppieren und validieren | 30.000 akzeptierte Szenarien, Coverage-Matrix, Schema-/Spannen-/Gruppentests, manuelle geschichtete Stichprobe; kein großer Lauf bei systematischen Fehlern. |
| 4 — Begrenzter Trainingsvergleich | Saubere Datenmischung gegen dieselbe Basis ohne neue Ergänzung vergleichen; feste Seeds und vergleichbares Schrittbudget | Verbesserung der P1-Domänen und schwächsten zulässigen Bereiche auf neuem Dev; Herkunftsgruppen getrennt. Schwache Metriken diagnostizieren statt nur Datenzahl erhöhen. |
| 5 — Skalierung 100.000 → 200.000 | Erst nach Phase 4 weitere Vorlagen, Schemata und seltene Wert-/Record-Konstellationen ergänzen; öffentliche Adapter parallel ausbauen | Je Stufe frische Audit-Stichprobe, Hash-/Dedup-/Coverage-Bericht und kurzer Dev-Vergleich. Öffentliche Mindestmenge separat nachweisen oder offen lassen. |
| 6 — Volltraining | Reproduzierbare Mischung nach Ausgabeformen und Domänen; Speicher-/Kontextprüfung; alle relevanten Modellköpfe trainieren | Gewählte Basis, Datenhashes, Rezept, Lernnachweise und Checkpoints dokumentiert. Auswahl ausschließlich anhand neuer Dev-Kriterien. |
| 7 — Abschlussprüfung | Separat kalibrieren, eingefrorenen Test genau einmal für den gewählten Kandidaten auswerten; API/SDK-Schemafunktion prüfen | Jede beworbene Domäne und Ausgabeform erfüllt ihre Gates; reproduzierbare Nutzungsbeispiele und ehrlicher Coverage-Bericht. Veröffentlichung bleibt ein eigener Schritt. |

### Auswahl der Trainingsbasis

Das abgeschlossene Generalistenmodell hat QA, Relationen und Klassifikation gelernt.
Eine Änderung der Dateien entfernt diesen Einfluss nicht aus den Gewichten.
Für den nächsten Hauptlauf eine dokumentierte Ausgangsbasis verwenden; zunächst
sauberes Training ab der ursprünglichen geeigneten GLiNER-Basis mit einer
Fortsetzung des bisherigen Modells in einem begrenzten Dev-Vergleich abwägen.
Eine Fortsetzung nicht als Training ausschließlich auf bereinigten Daten bezeichnen.
Der Plan verspricht kein nachträgliches „Unlearning“ und keine QA-freie Vortrainingshistorie.

### Ressourcen und Grenzen

- Ausschließlich lokale Entwicklung mit `nix develop`; keine Cloud-GPU-Aufträge.
- Externe Teacher-Kosten insgesamt höchstens 500 USD einschließlich bereits
  reservierter/verausgabter Beträge aus dem bestehenden Ledger.
- Vor jeder Generierungsstufe Tokenbedarf und Restbudget berechnen; nach einer
  kleinen bezahlten Stichprobe die Kosten pro akzeptiertem Beispiel aktualisieren.
- Vorwiegend lokale Fakten-/Textgeneratoren; LLM-Aufrufe gezielt für zusätzliche
  Renderfamilien und Paraphrasen. Nicht 200.000 ungeprüfte Teacher-Aufrufe bestellen.
- Fortsetzung nach Fehlern muss IDs und Kostenreservierungen erhalten; keine
  doppelten API-Kosten durch blindes Wiederholen.
- Bestehende Release-Gewichte nicht überschreiben, keine Uploads oder Sichtbarkeitsänderungen.

## 9. Definition of Done

Der Ausbau ist erst abgeschlossen, wenn:

1. Keine ausgeschlossene Aufgabe in den aktiven Trainings-/Evaluationsdaten oder Gates verbleibt.
2. Alle 16 Domänen konkrete Schemata, Datenherkunft und überprüfbare Coverage besitzen;
   nicht erfüllte Domänen ausdrücklich als nicht freigegeben gekennzeichnet sind.
3. Öffentliche Quelldaten, Annotationen und synthetische Ableitungen nachvollziehbar
   getrennt sind; das 200.000-Realziel nachgewiesen oder weiterhin offen ausgewiesen ist.
4. Die Mischung nicht mehr von NER oder wenigen Templates dominiert wird.
5. Vollständige Objekte und wiederholte Datensätze innerhalb der nachgewiesenen
   nativen beziehungsweise Pipeline-Kapazität funktionieren.
6. Die >70-%-Gates pro Fachdomäne und Ausgabeform mit unabhängiger Evaluation erfüllt sind.
7. Eine konkrete, getestete API/SDK-Nutzung für die unterstützten Schemata vorliegt;
   bisherige Quantity-API-Funktionalität allein belegt das nicht.


## Verbindliche Feldverträge und Reparaturtraining

Jedes normalisierte Feld deklariert Werttyp, Formatoptionen und Kardinalität als
Eingabeschema. Produktionscode darf diese Regeln nicht anhand von Gold-Antworten
oder Benchmarkergebnissen auswählen. Trainingsspannen und unabhängig annotierte
Endwerte werden getrennt gespeichert; fehlende Endwert-Annotationen bleiben als
Lücke sichtbar. Ursprüngliche Textbelege und Record-Grenzen bleiben erhalten.

Zusätzlich zu Domänen-/Form-Gates werden Feldrollen nach Herkunft getrennt geprüft:
Präzision, Recall, F1, positive Fallzahl und Fehlerbeispiele. Standardmäßig sind
mindestens 30 positive Annotationen und >70 % Feld-F1 erforderlich. Geringere
Fallzahlen oder unvollständige Annotationen belegen keine Zuverlässigkeit.

Reparaturtraining nutzt unabhängige Trainingsquellen und realistisch verfasste,
unabhängig geprüfte neue Prosa-Beispiele. Dev-Fehler dürfen Fehlerklassen bestimmen,
aber weder Dev-Texte noch ihre Umformulierungen gelangen ins Training. Native
Trainierbarkeit, Quellgruppen, exakte und nahe Heldout-Duplikate werden geprüft.
Mehrfachnutzung bleibt begrenzt; eine größere Zahl unabhängiger Quellen hat Vorrang.

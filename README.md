# gcloud-air-leakage-detection
Project enabling near real-time detection of Air Leakages for Fehrer


## Introduction

Ziel dieses Projekts ist der Aufbau einer In Memory Lösung zur frühzeitigen Erkennung von Luftlecks in Formträgern des Unternehmens Fehrer.
Grundlage für die Erkennung sind Daten, die von Fehrer bereitgestellt wurden. Diese beinhalten die Identifikation des Formteils selbst, die Arbeitsprogramme die es ausführt, den Timestamp, die Position in der Halle sowie den Druck im Formträger. Anhand dieser Merkmale soll in diesem Projekt ein KI-Modell gemäß dem Artikel "Time-Series Anomaly Detection Service at Microsoft" (Quelle: https://arxiv.org/pdf/1906.03821.pdf) zu trainiert werden, das später für die Erkennung der Luftlecks zuständig ist. Dieses Modell soll anschließend eine Warnung herausgeben, die von Ferher in ihrem Unternehmen in beliebiger Form angezeigt werden kann.

## Gcloud Alerting System

### PubSub

Da Fehrer bereits Dienstleistungen von Google im Einsatz hat wurde entschieden, auch in diesem Projekt Google einzusetzen. Für das Alerting System werden die Daten über die Pub/Sub Funktionalität der Google Cloud erfasst. Von dort aus werden die Daten in ein integriertes Docker-Netzwerk geladen. Dieses Netzwerk enthält Docker Container, die Funktionalitäten via Python, Kafka und Flink ausführen.

### Apache Flink in Dataproc

Apache Flink ist in diesem Projekt dafür zuständig, die Daten, die aus der Google Cloud via Pub/Sub geladen werden, in Fenster aufzuteilen. Dabei wurde sich hier für ein Tumbling Window von 30 Minuten entschieden. Dabei handelt es sich um die Zeit, die eine reguläre Pause in der Produktion von Fehrer dauert.

Für das Laden der Daten aus der Cloud nach Flink wurde sich hier entschieden, Kafka als Zwischenstation einzubinden. Dies hat den Vorteil, dass der Code in Python verfasst werden kann. Eine Alternative Lösung in Java oder Scala denkbar.

### AutoML (trained model)

Um die Luftlecks zu erkennen werden die Daten anschließend auf den Dienst AutoML von Google geladen. Dort wird das vortrainierte Modell gehostet. Hier werden die Daten auf Auffälligkeiten untersucht, gelabelt und anschließend zurück nach Flink geladen.

### Aggregated Database (Memorycloud)

Um entsprechend auf die Daten zugreifen zu können schreibt Flink die gelabelten Daten in eine In Memory Datenbank, die ebenfalls in der Cloud gehostet wird. Hier wurde sich für Redis entschieden.

### Alerting DataGrid API

Um eine Warnung zu erhalten wird für dieses Projekt eine regelmäßige Abfrage auf die In Memory Datenbank angedacht. Diese Abfrage liest für einen auszuwählenden Zeitraum aus, wie viele auffällige Datenpunkte in dem Datensatz vorhanden sind. Anhand des Verhältnisses zu den unauffälligen Datenpunkten wird entschieden, ob ein Luftleck vorliegt oder nicht. Zuletzt wird das Ergebnis wieder mittels Pub/Sub in die Google Cloud geschrieben und kann von den Systemen von Fehrer in das bestehende Alarmsystem eingebunden werden.

## Implementation as Docker Microservices

Kafka als Mittelschicht zwischen PubSub und Flink, weil PyFlink keine PubSub Connector besitzt.

### Apache Kafka


### Apache Flink


### Redis In-Memory





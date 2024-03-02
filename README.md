# gcloud-air-leakage-detection
Project enabling near real-time detection of Air Leakages for Fehrer

## Einführung

Ziel dieses Projekts ist der Aufbau einer In Memory Lösung zur frühzeitigen Erkennung von Luftlecks in Formträgern des Unternehmens Fehrer.
Grundlage für die Erkennung sind Daten, die von Fehrer bereitgestellt wurden. Diese beinhalten die Identifikation des Formteils selbst, die Arbeitsprogramme die es ausführt, den Timestamp, die Position in der Halle sowie den Druck im Formträger. Anhand dieser Merkmale soll in diesem Projekt ein KI-Modell gemäß dem Artikel "Time series anomaly detection with reconstruction-based state-space models" ([Quelle](https://arxiv.org/pdf/2303.03324.pdf)) zu trainiert werden, das später für die Erkennung der Luftlecks zuständig ist.

Dieses Modell soll anschließend die ankommende Datenpunkte als abnormal bzw. labeln. Basierend auf diesen Labeln wird eine Entscheidung getroffen, ob eine Warnung herausgeben ist. Die Warnungen kann dann Ferher in ihrem Unternehmen in beliebiger Form angezeigt werden kann.

Da Warnmeldungen in real-time geben sollen, haben wir einen Stream Processing Konzept entwickelt, der einen Microservice-Architektur abbildet. Dazu haben wir einen Diagram-Entwurf entwickelt, dass die Google Cloud Services mit sich einbezieht:

![GcloudAlertingSystem](https://github.com/vladislabv/gcloud-air-leakage-detection/assets/100155839/ad54447d-1f28-446c-8b2a-a187d361768f)

## Bidirectional Dynamic State-Space Model

Für das Training von BDM wurden die initialen Daten in folgende Struktur reingebracht:

- `timestamp`
- `air_Pressure` (Float-Werten im Bereich von 4 bis 8 bar)
- `ft_type` (FT1 bis FT8)

Die Training-Daten machen 75% Prozent des Datensatzes aus und diese wurden von Luftdruck-Leakagen bereining, da dies vom BDM verlangt ist. Das ist notwendig, damit das Modell sich ein "normales" Datenverlaufspattern merken kann. Das Training ist mit Validierung vorgesehen, was einen potentiellen Bias minimiert.

Der Test-Datensatz beinhaltet eine zusätzliche Metrik `label`, die als `1` die Leakagen bzw. allgemein das abnormale Verhalten markiert. Dementsprechend `0` steht für den normalen Verlauf vom Luftdruck.

Wir sind leider mit einem unerwartetem Ergebnis nach Training rausgekommen, was schlechter als Raten scheint:

![ROC Kurve](https://github.com/vladislabv/gcloud-air-leakage-detection/assets/100155839/3ae09605-4fb5-4bd0-bfd6-74d8bb2fbb7f)

Wir haben den Verdacht, dass der Datensatz zu kurz ist bzw. die Leakagen sind kaum in dem Testendaten vertreten, sodass das BDM hier ein bedauerndes Ergebnis geliefert hat.

## Google Cloud Alerting System

### Gooble Pub/Sub

Da Fehrer bereits Dienstleistungen von Google im Einsatz hat wurde entschieden, auch in diesem Projekt Google einzusetzen. Für das Alerting System werden die Daten über die Pub/Sub Funktionalität der Google Cloud erfasst.

Wir nehmen an, dass die jegliche aufgenommenen Events mit Luftdruck kommen auf einen einzelnen Topic `ft-sensor`. Alle Microservices haben hier nur read-only Zugriff und man geht davon aus, der Topic ist resistentner Datenspeicher von initialen Daten.

Der einzige Subscriber ist hier eine Apache Flink Instanz, die in `Dataproc-Service` läuft. Durch das Erhöhen von der Worker-Anzahl, kann das System leicht skaliert werden.

Die einfache Logik, wie Publishing und Subscribing sind in dem Repo von `gcloud/pubsub` Dateien vertreten.

### Apache Flink in Dataproc

Apache Flink ist in diesem Projekt dafür zuständig, die Daten, die aus der Google Cloud via Pub/Sub geladen werden, in Fenster aufzuteilen. Das machen wir, um eine Latenz von häufigen Abfragen von Flink an Kafka geringer zu halten. Deswegen haben wir uns für einen relativen kleinen Tumbling Time Window von etwa einer Minute entschieden. 

Flink implementiert folgende Logik als Stream Processor:

1. Batch wird erfasst
2. Events in Form von JSON encoded und in eine Liste gepackt
3. Die Liste wird an ein traininertes Modell via REST API verschickt
4. Die Antwort wird in eine Cloud Memorystore Instanz aggregiert geschrieben.

Die Logik ist noch WIP, und der angefangene Arbeit ist unter `docker/job/artifacts/kstream.py`.

### AutoML (trained model)

Um die Luftlecks zu erkennen werden die Daten anschließend auf den Dienst AutoML von Google geladen. Dort wird das vortrainierte Modell gehostet. Hier werden die Daten auf Auffälligkeiten untersucht, gelabelt und anschließend zurück nach Flink geladen.

### Aggregated Database

Um entsprechend auf die Daten zugreifen zu können schreibt Flink die gelabelten Daten in eine In Memory Datenbank, die ebenfalls in der Cloud gehostet wird. Hier wurde sich für Redis bzw. Memorycloud entschieden.

Da unsere Entscheidungsregel auf einer laufenden Verteilung von abnormalen Luftdruckwerten innerhalb einer Zeiteinheit basiert, müssen wir die eingehende gelabellte Datenpunkte nach dem `eventTime` eingrenzen. Dies möchten wir per Redis-Increments, dabei würde der Key durch die Angabe von FT Nummer und der Zeiteinheit (von-bis) eindeutig sein. Beispiel:

```
{
  "FT_1": {
    "2023-03-15T12:00:00-2023-03-15T12:30:00": 30
  }
}
```
Die Zahl 30 besagt, wie viele Datenpunkte innerhalb von 12 bis 12:30 am 15.03.2023 von dem Modell als "abnormal" gelabellt waren. In dem Fall haben wir uns auf 30 Minuten fokussiert (Tumbling Time Window), da es sich um die Zeit handelt, die eine reguläre Pause in der Produktion von Fehrer dauert.

### Alerting DataGrid API

Um eine Warnung zu erhalten wird für dieses Projekt eine regelmäßige Abfrage auf die In Memory Datenbank angedacht. Diese Abfrage liest für einen auszuwählenden Zeitraum aus, wie viele auffällige Datenpunkte in dem Datensatz vorhanden sind. Anhand des Verhältnisses zu den unauffälligen Datenpunkten wird entschieden, ob ein Luftleck vorliegt oder nicht. 

Durch die extreme Aggregation von Daten, erlauben wir uns mittels einer Abfrage die allen aggregierten Daten aus der In-Memory Datenbank zugreifen und daraus eine Standardabweichung `Std` schnell berechnen. Dieser Wert ist einen Merkmal, der einmalig mit noch nicht geprüften Ausprägungen geprüft wird. Dabei gilt der folgender Regel, sollte hier 30 > 3 * `Std` sein, wird der Zeitfenster für einen bestimmten FT als verdächtig gekennzeichet. 

Die Metadaten von verdächtigen von Zeitfenstern werden in einen Alert fließen, der über einen gesonderten Pub/Sub Topic abgewickelt wird. 

## Implementation as Docker Microservices

Kafka als Mittelschicht zwischen PubSub und Flink, weil PyFlink keine PubSub Connector besitzt.

Da wir als Team keine erforderliche Java-Vorkenntnisse besitzen, waren wir gezwungen, pyFlink zu nutzen. Die Python API zu Apache Flink ist leider nicht syncron zu allen neuen Flink-Features gehalten wird, so z.B. werden die Redis und Pub/Sub Connectors noch nicht unterstützt (Stand 2.3.2024).

Darüber hinaus, haben wir Apache Kafka als einen Vermittler von Google Pub/Sub zu Apache Flink genutzt. Das Microservicing haben wir mithilfe von Docker-Containers gebaut.

Dabei besteht unser Netzwerk aus einem Apache Kafka (+ Zookeeper), einem Producer und einem Apache Flink Container. Der Producer pusht die Messages in einen Kafka Topic, der von dem Flink gelesen werden kann. 

Die verbleibende Bausteine sind noch zu schaffen, wie Modell-Deployment mit FastAPI, Einrichtung von einem Redis-Cluster und einen Alerting Modul.

Jedoch die bestehende Arbeit ist unter `docker/docker-compose.yml` und `docker/Dockerfile-pyflink` zu finden.



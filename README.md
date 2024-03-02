# gcloud-air-leakage-detection
Project enabling near real-time detection of Air Leakages for Fehrer


## Introduction

Anfangen mit Anforderungen an das System: real-time processing (stream processing pipeline) mit alerting/notification modul


## Gcloud Alerting System

Hier einfach die Bausteine aus dem Diagramm beschreiben. Also, Anfang Google Pub/Sub, dann zu Flink in Dataproc...


### PubSub


### Apache Flink in Dataproc


### AutoML (trained model)


### Aggregated Database (Memorycloud)


### Alerting DataGrid API


## Implementation as Docker Microservices

Kafka als Mittelschicht zwischen PubSub und Flink, weil PyFlink keine PubSub Connector besitzt.

### Apache Kafka


### Apache Flink


### Redis In-Memory





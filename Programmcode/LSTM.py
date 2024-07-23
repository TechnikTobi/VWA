# Programmcode für LSTMs
# VWA: "Künstliche Neuronale Netzwerke und ihr Verhalten beim MNIST Datensatz"
# Autor: Tobias Prisching / 8C / 2018, 2019
# Betreuer: Mag. Christoph Hoedl
# Vorlage: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#          Zuletzt aufgerufen am 2018-09-09
# Nachschlagwerk: https://keras.io
# Geschrieben für Python 3.6.6, Keras 2.0.8, PlaidML 0.3.4

# Nötig für Vorbereitung der Daten
import numpy

# Keras Library zum Aufbau der Modelle
import keras

# Nötig für File-Management
import os
import time


# MNIST Daten werden in Trainings- und Testbeispiele eingeteilt
# (Trainingsbeispiele beinhalten Validationsbeispiele)
(xTrainingDaten, yTrainingDaten), (xTestDaten, yTestDaten) = keras.datasets.mnist.load_data()

# Anpassung der Dimensionalität der Daten
xTrainingDaten = xTrainingDaten.reshape(xTrainingDaten.shape[0], 28, 28)
xTestDaten = xTestDaten.reshape(xTestDaten.shape[0], 28, 28)

# Umwandlung der Datentypen der Beispiele in Float, um diese anschließend
# durch 255 zu dividieren
xTrainingDaten = xTrainingDaten.astype("float32")
xTestDaten = xTestDaten.astype("float32")

# Dividieren der Daten durch 255 um diese in einen Intervall zwischen 0 und 1
# zu bringen
xTrainingDaten /= 255
xTestDaten /= 255

# Umwandlung der gewünschten Ausgaben in 10-dimensionale Vektoren, eine
# Dimension pro Ziffer
yTrainingDaten = keras.utils.np_utils.to_categorical(yTrainingDaten, 10)
yTestDaten = keras.utils.np_utils.to_categorical(yTestDaten, 10)


# Erstellen des Ordners für alle von diesem Programm erstellten Datein
ordnerNameZeit = round(time.time())
os.makedirs("LSTM"+str(ordnerNameZeit))
os.chdir("./LSTM"+str(ordnerNameZeit))

# Dictonary welches den Index der falsch erkannten Ziffern und von wie vielen
# Netzen diese falsch erkannt wurden speichert
falschErkannteZiffernVonAllen = {}
for i in range(10000):
    falschErkannteZiffernVonAllen[i]=0


# Kopfzeile für die .csv Datei mit den gesamelten Ergebnissen
csvErgebnisseHeader = "Index-Nr;Modell-Nr;Aktf.;rek. Aktf.;Bias;Lernrate;Mini Batch;Epochs;Trainingszeit (s);Train-Cost;Val.-Cost;Test-Cost;Train-Quote;Val.-Quote;Test-Quote\n"

# Kopfzeile für die .csv Datei mit jenen Ziffern der Test Menge, welche von allen
# Netzen falsch erkannt wurden
csvFalschErkanntVonAllenHeader = "Ziffer-Index;Falsch erkannt von\n"

# Kopfzeile für die .csv Datei mit jenen Ziffern der Test Menge, welche von einem
# bestimmten Netz falsch erkannt wurden und was stattdessen seine Antwort hat
csvFalschErkanntEinzelnHeader = "Ziffer-Index;Erkannt als;Richtige Antwort:\n"

# Öffnen dieser .csv Datei und Einfügen des Headers
csvMitErgebnissen = open("ergebnisse.csv", "a")
csvMitErgebnissen.write(csvErgebnisseHeader)


class LSTM():

    # Initialisierung eines LSTMs
    def __init__(self,
                 hiddenLayerAufbau,
                 aktivierungsfunktion,
                 rekurrenteAktivierungsfunktion,
                 bias,
                 lernrate,
                 miniBatchGroesse,
                 indexNr,
                 hiddenLayerAufbauNr):

        # Speichern der übergebenen Argumente
        self.hiddenLayer = hiddenLayerAufbau
        self.aktivierungsfunktion = aktivierungsfunktion
        self.rekurrenteAktivierungsfunktion = rekurrenteAktivierungsfunktion
        self.bias = bias
        self.lernrate = lernrate
        self.miniBatchGroesse = miniBatchGroesse
        self.indexNr = indexNr
        self.hiddenLayerAufbauNr = hiddenLayerAufbauNr

        # Setzen des Seeds des Zufallsgenerators, um die Ergebnisse der Experimente
		# reproduzieren zu können
		# Hierfür wird die Index-Nr. des KNNs verwendet
        numpy.random.seed(self.indexNr)

        # Es handelt sich um ein sequenzielles Model
        self.modell = keras.models.Sequential()

        # Hinzufügen der Hidden Layer zum Modell
        for i, hiddenLayer in enumerate(hiddenLayerAufbau):
            if hiddenLayer[0][0] == "LSTM":
                if i == 0:
                    self.modell.add(keras.layers.LSTM(hiddenLayer[1][0],
                                                      activation=aktivierungsfunktion,
                                                      recurrent_activation=rekurrenteAktivierungsfunktion,
                                                      return_sequences=hiddenLayer[2][0],
                                                      input_shape=(28,28),
                                                      implementation=2,
                                                      unroll=True,
                                                      use_bias=self.bias))
                else:
                    self.modell.add(keras.layers.LSTM(hiddenLayer[1][0],
                                                      activation=aktivierungsfunktion,
                                                      recurrent_activation=rekurrenteAktivierungsfunktion,
                                                      return_sequences=hiddenLayer[2][0],
                                                      implementation=2,
                                                      unroll=True,
                                                      use_bias=self.bias))

            elif hiddenLayer[0][0] == "Flatten":
                self.modell.add(keras.layers.Flatten())

            else:
                if i == 0:
                    self.modell.add(keras.layers.Dense(hiddenLayer[1][0],
                                                      activation=aktivierungsfunktion,
                                                      input_shape=(28,28),
                                                      use_bias=self.bias))
                else:
                    self.modell.add(keras.layers.Dense(hiddenLayer[1][0],
                                                      activation=aktivierungsfunktion,
                                                      use_bias=self.bias))

        # Hinzufügen des Output Layers zum Modell
        self.modell.add(keras.layers.Dense(10, activation="softmax"))

        # Initialisierung der Optimierungs-Methode
        stochasticGradientDescent = keras.optimizers.SGD(lr=self.lernrate,
                                   decay=0,
                                   momentum=0,
                                   nesterov=False)

        # Konfigurieren des Lernprozesses
        self.modell.compile(loss="mean_squared_error",
                           optimizer= stochasticGradientDescent,
                           metrics=["accuracy"])


    # Funktion fürs Trainieren und Testen des LSTMs
    def trainierenUndTesten(self):

        os.makedirs(str(self.indexNr))
        os.chdir(str(self.indexNr))

        # Implementierung des Early Stoppings
        earlyStopping = keras.callbacks.EarlyStopping(monitor="val_acc",
                                                      min_delta=0,
                                                      patience=5,
                                                      verbose=1, mode="auto")

        # Speichern der besten Parameter
        modelCheckpoint = keras.callbacks.ModelCheckpoint("parameters-"+str(self.indexNr)+".hdf5",
                                                          monitor="val_acc",
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode="auto",
                                                          period=1)

        # Speichern der Loss- und Accuracy-Werte nach jedem Epoch
        csvLogger = keras.callbacks.CSVLogger("trainingLog-"+str(self.indexNr)+".csv", separator=";", append=False)

        # Liste aller nicht standardmäßigen Callbacks
        callbacksListe = [modelCheckpoint, earlyStopping, csvLogger]


        # Trainieren des KNNs und Messung der dafür benötigten Zeit
        startZeit = time.time()
        trainingHistory = self.modell.fit(xTrainingDaten,
                                    yTrainingDaten,
                                    batch_size=self.miniBatchGroesse,
                                    epochs=3,
                                    verbose=1,
                                    callbacks=callbacksListe,
                                    validation_split=1/6)

        # Berechnung der fürs Trainieren benötigten Zeit
        # in Sekunden
        trainingszeit = round(time.time() - startZeit, 1)


        # Laden der besten Parameter
        self.modell.load_weights("parameters-"+str(self.indexNr)+".hdf5", by_name=False)

        # Testen des KNNs auf die verschiedenen Beispiele
        trainingQuote = self.modell.evaluate(xTrainingDaten[:50000], yTrainingDaten[:50000], verbose=1)
        validationQuote = self.modell.evaluate(xTrainingDaten[50000:], yTrainingDaten[50000:], verbose=1)
        testQuote = self.modell.evaluate(xTestDaten, yTestDaten, verbose=1)

        # Speichern der inkorrekt erkannten Ziffern
        falschErkannteZiffernEinzeln = []
        testAntwortKNN = self.modell.predict(xTestDaten, verbose=0)
        for i in range (10000):
            if numpy.argmax(testAntwortKNN[i]) != numpy.argmax(yTestDaten[i]):
                falschErkannteZiffernEinzeln.append([i, numpy.argmax(testAntwortKNN[i]), numpy.argmax(yTestDaten[i])])
                falschErkannteZiffernVonAllen[i]+=1
        csvFalschErkanntEinzeln = open("csvFalschErkannt-" + str(self.indexNr) + ".csv", "a")
        csvFalschErkanntEinzeln.write(csvFalschErkanntEinzelnHeader)
        for falschErkannteZiffer in falschErkannteZiffernEinzeln:
            csvFalschErkanntEinzeln.write(str(falschErkannteZiffer[0]) + ";" + str(falschErkannteZiffer[1]) + ";" + str(falschErkannteZiffer[2]) + "\n")
        csvFalschErkanntEinzeln.close()


        # Speichern des Modells als .json Datei und der Parameter als .h5 (für
        # spätere Umwandlung in CoreML Dateien)
        jsonDatei = open("model-"+str(self.indexNr)+".json", "w")
        jsonDatei.write(self.modell.to_json())
        jsonDatei.close()
        self.modell.save_weights("modelWeights-"+str(self.indexNr)+".h5")

        os.chdir("./..")

        # Zurückgeben der Ergebnisse
        returnString = str(self.indexNr) + ";" + str(self.hiddenLayerAufbauNr) + ";" + self.aktivierungsfunktion + ";" + self.rekurrenteAktivierungsfunktion + ";" + str(self.bias) + ";" + str(self.lernrate) + ";" + str(self.miniBatchGroesse) + ";" + str(len(trainingHistory.history["acc"])) + ";" + str(trainingszeit) + ";" + str(round(trainingQuote[0],4)) + ";" + str(round(validationQuote[0],4)) + ";" + str(round(testQuote[0],4)) + ";" + str(round(trainingQuote[1],4)) + ";" + str(round(validationQuote[1],4)) + ";" + str(round(testQuote[1],4)) + "\n"
        return(returnString)



# Index zur Identifizierung der verschiedenen LSTMs
index = 1

# Erstellen der Array mit den verschiedenen Modellen:
modell1 = [[["LSTM"],[10],[False]]]

modell2 = [[["LSTM"],[10],[True]],
           [["LSTM"],[10],[False]]]

modell3 = [[["Dense"],[10]],
           [["LSTM"],[10],[False]]]

modell4 = [[["Dense"],[10]],
           [["LSTM"],[10],[True]],
           [["LSTM"],[10],[False]]]

modelle = [modell1,modell2,modell3,modell4]

# Schleifen zum Testen der verschiedenen Kombinationen
for hiddenLayerAufbauNr, hiddenLayerAufbau in enumerate(modelle):
    for lernrate in [1,0.1,0.01]:
        for miniBatchGroesse in [8,32,128]:
            for aktivierungsfunktion in ["sigmoid","tanh"]:
                for bias in [True]:
                    for rekurrenteAktivierungsfunktion in ["sigmoid","tanh"]:
                        # Erstellung eines LSTMs mit dieser Kombination an
                        # Hyperparametern
                        KNN = LSTM(hiddenLayerAufbau,
                                  aktivierungsfunktion,
                                  rekurrenteAktivierungsfunktion,
                                  bias,
                                  lernrate,
                                  miniBatchGroesse,
                                  index,
                                  hiddenLayerAufbauNr+1)

                        # Testen des LSTMs und Niederschreiben der Ergebnisse
                        csvMitErgebnissen.write(KNN.trainierenUndTesten())
                        csvMitErgebnissen.close()
                        csvMitErgebnissen = open("ergebnisse.csv", "a")

                        # Inkrementierung des Index
                        index+=1

# Öffnen, speichern und schließen der .csv Datei, welche speichert, welche der
# Testbeispiele von wie vielen Netzen falsch erkannt wurden
csvFalschErkanntVonAllen = open("falschErkannteZiffernVonAllen.csv", "a")
csvFalschErkanntVonAllen.write(csvFalschErkanntVonAllenHeader)
for ziffer in falschErkannteZiffernVonAllen:
    csvFalschErkanntVonAllen.write(str(ziffer)+";"+str(falschErkannteZiffernVonAllen[ziffer])+"\n")
csvFalschErkanntVonAllen.close()

# Schließen der Datei mit den Ergebnissen, verlassen des Ordners der LSTMs
csvMitErgebnissen.close()
os.chdir("./..")

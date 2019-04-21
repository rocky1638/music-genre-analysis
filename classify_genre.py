import sys
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import extract_features
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class color:
    PURPLE = '\033[95m'
    PINK = '\033[35m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    END = '\033[0m'


genres = pd.read_csv('fma_metadata/genres.csv')['title'].values


def printResults(predictedGenres):
    file = open('title.txt', 'r')
    title = file.read()
    print(f"{color.PINK}{title}{color.END}")

    print(f"{color.PINK}** ------------------------------------------- **{color.END}")
    print(
        f"{color.BOLD}Your song's most likely genre was: {color.RED}{predictedGenres[0]}{color.END}.{color.END}")
    print(
        f"Your song's second most likely genre was: {color.YELLOW}{predictedGenres[1]}{color.END}.")
    print(
        f"Your song's third most likely genre was: {color.BLUE}{predictedGenres[2]}{color.END}.")
    print(f"{color.PINK}** ------------------------------------------- **{color.END}")


def main():
    filepath = sys.argv[1]
    model = keras.models.load_model('final_model.h5')
    sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)

    model.compile(
        optimizer=sgd_optimizer,
        loss='kullback_leibler_divergence',
        metrics=['accuracy']
    )

    feature_vector = extract_features.compute_features(filepath)
    fitted_scaler = joblib.load('scaler.save')

    scaled_vector = fitted_scaler.transform(
        feature_vector.values.reshape(1, -1)
    )

    prediction = model.predict(scaled_vector)[0]
    maxIndices = []
    maxIndices.append(np.argmax(prediction))

    prediction[maxIndices[0]] = -1
    maxIndices.append(np.argmax(prediction))

    prediction[maxIndices[1]] = -1
    maxIndices.append(np.argmax(prediction))

    predictedGenres = [genres[i] for i in maxIndices]

    printResults(predictedGenres)


if __name__ == '__main__':
    main()

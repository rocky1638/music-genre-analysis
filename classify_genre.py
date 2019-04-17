import sys
import numpy as np
import tensorflow.keras as keras
import extract_features
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


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

    print(maxIndices)


if __name__ == '__main__':
    main()

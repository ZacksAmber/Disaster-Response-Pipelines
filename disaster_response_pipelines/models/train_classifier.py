import sys
import joblib

from NeuralNetwork import NeuralNetwork


# Reference: https://stackoverflow.com/questions/49621169/joblib-load-main-attributeerror
# Seperarte model dump code and model training code
# for avoding AttributeError when using joblib load model.
def main():
    if len(sys.argv) == 4:
        database, table, model_filename = sys.argv[1:]
        model = NeuralNetwork().export_model(database, table)

        print(f'Saving model...\n    MODEL: {model_filename}')
        joblib.dump(model, model_filename)
        print('Trained model saved!')

    else:
        print('\nPlease provide the filename of the disaster messages database and table '
              'as the first and second arguments and the filename of the pickle file to '
              'save the model to as the second argument. \n\nExample: python3 '
              'train_classifier.py disaster_response.db disaster_response classifier.pkl')


if __name__ == '__main__':
    main()

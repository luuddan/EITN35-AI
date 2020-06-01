import os

def print_to_file(acc, val_acc, loss, val_loss, epochs, layers, model):
    # Create "training_results" folder if it does not exist

    os.chdir("C:/Users/eitn35/Documents/EITN35_EVOLVE/models_and_weights_EVOLVE/models")

    try:
        if not os.path.exists('training_results'):
            os.makedirs('training_results')
    except OSError:
        print('Error: Creating directory of data')

    results = "C:/Users/eitn35/Documents/EITN35_EVOLVE/models_and_weights_EVOLVE/models/training_results/"

    os.chdir(results)
    current_results = os.listdir(results)
    current_results.sort()
    working_file = current_results[len(os.listdir(results)) - 1]
    print("Latest file is: " + str(working_file))

    file = open(working_file, "a+")
    file.write("\n \n -----------------------------------")
    file.write("\n Model " + str(model) + " Run Result")
    file.write("\n -----------------------------------")
    file.write("\n Hidden Layers:           " + str(layers))
    file.write("\n Epochs:                  " + str(epochs))
    file.write("\n Training Set Accuracy:   " + str(acc))
    file.write("\n Validation Set Accuracy: " + str(val_acc))
    file.write("\n Training Loss:           " + str(loss))
    file.write("\n Validation Loss:         " + str(val_loss))
    file.close()
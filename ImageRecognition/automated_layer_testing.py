import os

#project file directory
work_dir = 'C:/Users/eitn35/PycharmProjects/EITN35-AI/ImageRecognition/'
os.chdir(work_dir)

#results directory, will be created if it non-existent
results = work_dir + 'training_results/'

# Create "training_results" folder if it does not exist
try:
    if not os.path.exists('training_results'):
        os.makedirs('training_results')
except OSError:
    print('Error: Creating directory of data')

file_name = 'temp'

def new_print_file(no_layers):
    #retrieves number of previous runs
    runNo = len(os.listdir(results))+1
    file_name = 'Model_' + str(no_layers) + '.txt'

    #creates new results file
    os.chdir(results)
    file = open(file_name, "w")
    file.write("Model " + str(no_layers))
    file.write("\n")
    file.close()
    os.chdir(work_dir)


def new_confusion_file():
    no_confusion = 0

    # creates new results file
    os.chdir(results)
    for i in range(len(os.listdir(results))):
        if("Confusion" in os.listdir(results)[i]):
            no_confusion += 1

    order = no_confusion+1
    file_name = 'Confusion_' + str(order) + '.txt'
    file = open(file_name, "w")
    file.write("Confusion Matrix File " + str(order))
    file.write("\n")
    file.close()
    os.chdir(work_dir)


#trains and tests the models with different layer configurations, writes acc, acc_loss, loss and val_loss to file
os.chdir(work_dir)

#learning rate, drop rate, regularization rate, run number, data fraction, batch size

#
#
# new_print_file(22)
#
# os.system("python Model_layers_2.py 0.00005 0 0 0 1 32")
new_print_file(102)
new_confusion_file()
os.system("python Model_layers_2.py 0.001 0 0.0005 1 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 2 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 3 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 4 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 5 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 6 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 7 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 8 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 9 1 32")
os.system("python Model_layers_2.py 0.001 0 0.0005 10 1 32")

# os.system("python Model_layers_2.py 0.005 0 0 18 1 32")
# os.system("python Model_layers_2.py 0.001 0 0 19 1 32")
# os.system("python Model_layers_2.py 0.0005 0 0 20 1 32")
# os.system("python Model_layers_2.py 0.0001 0 0 21 1 32")
# os.system("python Model_layers_2.py 0.0005 0 0 22 1 32")
#
# os.system("python Model_layers_2.py 0.001 0 0.005 23 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.001 24 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.0005 25 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.0001 26 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.0001 27 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.0001 28 1 32")
# os.system("python Model_layers_2.py 0.001 0 0 29 1 32")
# os.system("python Model_layers_2.py 0.001 0 0 30 1 32")
# os.system("python Model_layers_2.py 0.001 0 0 31 1 32")

# new_print_file(104)
#
# os.system("python Model_layers_2.py 0.0001 0 0 1 1 32")
# os.system("python Model_layers_2.py 0.0005 0 0 2 1 32")
# os.system("python Model_layers_2.py 0.001 0 0 3 1 32")
# os.system("python Model_layers_2.py 0.01 0 0 4 1 32")
#
# os.system("python Model_layers_2.py 0.001 0.1 0 5 1 32")
# os.system("python Model_layers_2.py 0.001 0.2 0 6 1 32")
# os.system("python Model_layers_2.py 0.001 0.25 0 7 1 32")
# os.system("python Model_layers_2.py 0.001 0.3 0 8 1 32")
# os.system("python Model_layers_2.py 0.001 0.4 0 9 1 32")
# os.system("python Model_layers_2.py 0.001 0.5 0 10 1 32")
#
# os.system("python Model_layers_2.py 0.001 0 0.0005 11 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.001 12 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.005 13 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.01 14 1 32")
# os.system("python Model_layers_2.py 0.001 0 0.1 15 1 32")




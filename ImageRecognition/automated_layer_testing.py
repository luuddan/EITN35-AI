import os

#project file directory
work_dir = 'C:/Users/eitn35/PycharmProjects/EITN35/'
os.chdir(work_dir)

#results directory, will be created if it non-existent
results = work_dir + '/training_results/'

# Create "training_results" folder if it does not exist
try:
    if not os.path.exists('training_results'):
        os.makedirs('training_results')
except OSError:
    print('Error: Creating directory of data')

#retrieves number of previous runs
runNo = len(os.listdir(results))+1
file_name = 'EVOLVE_Baseline_model_' + str(runNo) + '.txt'

#creates new results file
os.chdir(results)
file = open(file_name, "w")
file.write("EVOLVE_Baseline_model_" + str(runNo))
file.write("\n")
file.close()

#trains and tests the models with different layer configurations, writes acc, acc_loss, loss and val_loss to file
os.chdir(work_dir)
##                                          L2 Drop Run_number
os.system("python Baseline_model_V3_224x224.py 0 0 1")
os.system("python Baseline_model_V3_224x224.py 0 0 2")
os.system("python Baseline_model_V3_224x224.py 0 0 3")
os.system("python Baseline_model_V3_224x224.py 0 0 4")
os.system("python Baseline_model_V3_224x224.py 0 0 5")
os.system("python Baseline_model_V3_224x224.py 0 0 6")
os.system("python Baseline_model_V3_224x224.py 0 0 7")
os.system("python Baseline_model_V3_224x224.py 0 0 8")
os.system("python Baseline_model_V3_224x224.py 0 0 9")
os.system("python Baseline_model_V3_224x224.py 0 0 10")

#Open the file back and print the contents
os.chdir(results)
print("Trying to open " + file_name)
f = open(file_name, "r")
if f.mode == 'r':
    contents =f.read()
    print(contents)
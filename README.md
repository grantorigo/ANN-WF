# ANN-WF

The ANN-WF repository gives the opportunity to investigate the ability of Artificial Neural Networks to approximate ground-state quantum wave functions. The ANN-multi-test.py script allows to test many different networks on the example of the 1D antiferromagnetic Heisenberg Hamiltonian. The file ANN-multi-test.py allows three types of batch gradient descent variants, but this has to be changed in the script variants. As standard method AdaGrad (see http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants for further information) is used. For now only batch GD is implemented, perhaps it might be extended to mini batch GD at a later point. The functionality of ANN-multi-test.py is best explained by walking thorught he input file, which we will do now.

## Config.json

ANN-multi-test.py needs an input file (Config.json) where one can specifiy the Fock space, the Hamiltonian, the way of testing and the network itself. Hereunder we can see an example for Config.json:
```Java
{
	"System":{
		"N": 8,
		"TotalSz": "0",
		"SignTransform": true
	},
	"Test":{
		"L_max": [11,11],
		"L_min": [9,9],
		"Epochs": 1e3,
		"Steps" : 1,
		"Precision": 8e-16,
    "Repetitions": 5,
    "Pre-Training": true
	},
	"Network":{
		"Name": "TDT",
		"Architecture":["Linear",0,"Tanh","Linear",1,"Triangle","Linear",0,"Tanh"],
		"Loss": "Energy"
	}
}
```
We begin with the settings of the system. First of all we set the system size N to any integer larger 0, but for the sake of your system memory you should not exceed "N": 16. Next one can restric the Fock space to the TotalSz = 0 subspace by setting "TotalSz": "0", anything else yields the whole Fock space. Finally the Heisenberg Hamiltonian 


## Output

## Plot

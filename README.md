# ANN-WF

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

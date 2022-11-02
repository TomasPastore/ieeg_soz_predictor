

# Summary:

Tool to analyse HFO data.  
Are HFOs good predictors of the SOZ to help refractory epilepsy treatment ?

# Dependencies 
	Install python packages detailed in requirements.txt in project root
        
        pip install -r requirements.txt

# Run
    From project root you can execute the following commands in shell:
    
    * interactive mode: python src/main.py -i
    * specific driver.py function previously set in main() function: python src/main.py  
    
# Project layout:
<pre>
ieeg_soz_predictor.
├── docs
│   ├── bibliography
│   ├── db_schema
│   ├── tesis
├── figures
│   ├── 1_global_data
│   └── 2_stats
│   └── 3_rate_soz_predictor_baselines
│   └── 4_ml_hfo_classifiers
├── README.md
├── requirements.txt
└── src
    ├── conf.py                                   # Definitions and global vars
    ├── data_dimensions.py                  # First queries and data dimensions
    ├── db_parsing.py                      # Database parsing and normalization
    ├── driver.py                       # Abstraction to navigate the main code
    ├── electrode.py                                          # Electrode class
    ├── event.py                                                  # Event class
    ├── graphics.py                                        # Figures generation
    ├── main.py                                 # Program execution entry point
    ├── ml_algorithms.py            # Sklearn machine learning algorithms calls
    ├── ml_hfo_classifier.py     # Sklearn for classifying pHFOs using features
    ├── partition_builder.py       # Making partitions of patients for crossval
    ├── patient.py                                              # Patient class
    ├── remove_artifacts. # Filters FRonO 300 HZ and RonO 180 HZ elec artifacts
    ├── soz_predictor.py               # Event-rate baseline per location, type
    ├── stats.py                              # HFO rate and feature stats code
    ├── utils.py                                                 # Useful code
    └── validation_names_by_loc.json      # Dict with random validation patient
</pre>

# Developed by 

**Tomás Pastore** <tpastore@dc.uba.ar> in collaboration with *Diego F. Slezak* and *Shennan A. Weiss* at **LIAA U.B.A** (2020).


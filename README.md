# SULI_fermilab_M4_analysis
Code created by Trevor Loe as a 2022 SULI intern at Fermilab under Dr. Diktys Stratakis. This code is developed for data analysis of the first commisioning of the Fermilab Muon Campus M4 line in May 2022. Actual data is not included as that is property of Fermi National Accelerator Laboratory.

## Paper
A paper was published based on the analysis done by this code on April 5, 2023, which can be found [here](https://iopscience.iop.org/article/10.1088/1748-0221/18/04/P04005).

## G4beamline files
The G4beamline folder contains all files used to run and analyze the G4beamline simulation. G4beamline is a free Geant4-based simulation; more info on it can be found [here](https://www.muonsinc.com/Website1/G4beamline). For this project, it was run from the command line. The ```.g4bl``` files used to run the simulation have been removed as they are the property of Fermilab. Data from the simulations is stored in the text files (VD_Diagnostic for the virtual detectors and MW___ for the real detectors simulated).

The folder baseline contains the files generated from running the simulation with all the magnets at nominal settings, getting readings from every detector

The folder diff_emit_run contains files generated from running the simulation with all magnets at nominal settings, but with different starting emittance values. This was test whether using a starting emittance value more consistant with the measured emittance would yield better agreement. 

The folder q930_sim contains the data from running the simulation at a different current value at magneti Q930 for testing purposes.

The folder quad_scan_sim contains the data generated from simulating the quad scan process: where a single magnet's current is slowly changed and the distribution of the beam is measured at a certain distance away. This is done automatically using the script [quad_scan_sim](G4beamline/quad_scan_sim/quad_scan_sim.sh), a shell script that is written for a linux operating system. It uses a loop to run the g4beamline simulation repeatedly, changing the current each time using the python script [change_I](G4beamline/quad_scan_sim/change_I.py). The resulting data is then saved into one of the sample_ folders, each named with their respective current value.

Each of the folders starting with three_screen_ contains the data generated in simulating the 3-screen process. Each is named with the magnet at the start of the beamline considered for the 3-screen process.

## Figures and Results
Histograms for the baseline simulation (all magnets are nominal currents) can be found within G4beamline/baseline/histograms. 

Figures generated for the simulated 3-screen method can be found within G4beamline/three_screen_figures.

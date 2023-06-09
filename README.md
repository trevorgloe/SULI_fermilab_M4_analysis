# SULI_fermilab_M4_analysis
Code created by Trevor Loe as a 2022 SULI intern at Fermilab under Dr. Diktys Stratakis. This code is developed for data analysis of the first commisioning of the Fermilab Muon Campus M4 line in May 2022. Actual data is not included as that is property of Fermi National Accelerator Laboratory.

Because the data has been removed, this code will not run as-is. New data would have to be provided and changed at any point in which the datafiles are referenced. 

## Paper
A paper was published based on the analysis done by this code on April 5, 2023, which can be found [here](https://iopscience.iop.org/article/10.1088/1748-0221/18/04/P04005).

## G4beamline files
The G4beamline folder contains all files used to run and analyze the G4beamline simulation. G4beamline is a free Geant4-based simulation; more info on it can be found [here](https://www.muonsinc.com/Website1/G4beamline). For this project, it was run from the command line. The ```.g4bl``` files used to run the simulation have been removed as they are the property of Fermilab. Data from the simulations is stored in the text files (VD_Diagnostic for the virtual detectors and MW___ for the real detectors simulated). The data within each G4beamline generated text file includes the particle type, position and velocity and time of interaction for a simulated particle hitting the screen. Using standard numpy libraries, the standard deviation can be extracted.

The folder baseline contains the files generated from running the simulation with all the magnets at nominal settings, getting readings from every detector

The folder diff_emit_run contains files generated from running the simulation with all magnets at nominal settings, but with different starting emittance values. This was test whether using a starting emittance value more consistant with the measured emittance would yield better agreement. 

The folder q930_sim contains the data from running the simulation at a different current value at magneti Q930 for testing purposes.

The folder quad_scan_sim contains the data generated from simulating the quad scan process: where a single magnet's current is slowly changed and the distribution of the beam is measured at a certain distance away. This is done automatically using the script [quad_scan_sim](G4beamline/quad_scan_sim/quad_scan_sim.sh), a shell script that is written for a linux operating system. It uses a loop to run the g4beamline simulation repeatedly, changing the current each time using the python script [change_I](G4beamline/quad_scan_sim/change_I.py). The resulting data is then saved into one of the sample_ folders, each named with their respective current value.

Each of the folders starting with three_screen_ contains the data generated in simulating the 3-screen process. Each is named with the magnet at the start of the beamline considered for the 3-screen process.

## mu2e-m4-mw-study-2022-05-25 files
The mu2e-m4-mw-study-2022-05-25 folder contains the files and scripts used for analyzing and generating data from the first commisioning of the M4 line. The data has been removed but the data used was in the form of a csv file and any similarly formatted file would work. 

Nearly all sub-directories of this folder contain figures generated from the scripts ([ellipse_plots](mu2e-m4-mw-study-2022-05-25/ellipse_plots.py), [base_prof](mu2e-m4-mw-study-2022-05-25/base_prof.py), etc.)

Each of the 3-screen data files contain many versions of the nonlinear fit done for the 3-screen method (each done with both scipy and GEKKO). These were done many times with standard deviations within the uncertainty of the fitted values to test for convergence. 

[th_twist_prop](mu2e-m4-mw-study-2022-05-25/th_twist_prop.py) (slight typo in the name) generates the tranformation matrices for the beam going through the beamline. It then uses the matrices to propogate the Twiss parameters for the beam as it goes down the beamline. This served as a theoretical basis for comparing the results of the data.

The main files used to perform the analysis are [quad_scan_data1](mu2e-m4-mw-study-2022-05-25/quad_scan_data1.py), [quad_scan_data2](mu2e-m4-mw-study-2022-05-25/quad_scan_data2.py), [new_three_screen1](mu2e-m4-mw-study-2022-05-25/new_three_screen1.py), [new_three_screen2](mu2e-m4-mw-study-2022-05-25/new_three_screen2.py) and [new_three_screen3](mu2e-m4-mw-study-2022-05-25/new_three_screen3.py) which perform the quad-scan and 3-screen method, respectively. While the 3-screen method saves the figures for each screen as it runs, the quad-scan method scripts only display two graphs when the script is finished. It is up to the user to save this figure.

## Analysis Scripts
All functions used for analysis are stored within the [plot_fnc](mu2e-m4-mw-study-2022-05-25/plot_fnc.py) script, for which many copies can be found throughout the code. The copies were to provide ease of access for import the libraries. The plot_fnc scripts contains the following important functions among many others:
- read_data: reads the text files generated by G4beamline containing simulated particle interactions, converting the data to a pandas dataframe
- read_scan_beamdata: reads the datafile created during the first M4 beamline comissioning. Creates a pandas dataframe from the data
- make_scan_beam_scatterplot: uses the dataframe created by the read_scan_beamdata function to create a scatterplot showing the particles intensities over the position along the screen. 
- make_hist_dir: creates a directory to save data_histograms, created as a folder in the current working directory
- extract_rowx_scan: extracts a specified row of data from the dataframe generated by the read_scan_beamdata function. This has to be done carefully as there are several rows of specifying data.
- guass_fit_wcuttoff: uses the function new_guass_fit to fit a guassian function to the intensity over position graph provided from the data files. It implements a cutoff for the data to get rid of noise far off of the center of gaussian. The lcut and rcut parameters passed in control this and are important in getting a valid guassian fit. 
- comp_3sc_uncert: takes in the guassian fit parameters as well as their uncertainty to generate the range of possible twiss parameters that the uncertainty in the standard deviations could generate. This is done by using both the thrsc and thrsc_gek functions (utilizing scipy and GEKKO's nonlinear equation fitting respectively) to do the three-screen method on the 8 different combinations of the extremal low and extremal high standard deviations given. Each call to the three-screen functions involves the convergence of the nonlinear fit, so the convergence for the method is printed each time. After all of these runs, the lowest and highest value for each twiss parameter is found, which can be reported as the uncertainy in the twiss parameters. 
- new_quad_scan____ each function with a different magnet's name on the end corresponds to the specific quadrupole magnet whose current was changed over time for the quad scan procedure. Each function uses the standard deviations squared (passed in) and the different magnetic focussing strengths to fit a parabola. Then generates the twiss parameters from the fitted parameters of the parabola. 

## Figures and Results
Histograms for the baseline simulation (all magnets are nominal currents) can be found within G4beamline/baseline/histograms. 

Figures generated for the simulated 3-screen method can be found within G4beamline/three_screen_figures.

Results from all the simulated 3-screen methods can be found in [three_screen_results](G4beamline/three_screen_results.csv), [all_three_screen_sim_results](G4beamline/all_three_screen_sim_results.xlsx). The standard deviations of the distributions for the case of nominal magnet currents is in [allmeanstdg4bl](allmeanstdg4bl.txt). 

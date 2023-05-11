#!/bin/bash
echo "Simulating quad-scan in G4beamline"
# loop to run through all currents
for I in -0.2678780586221703 -0.472784859302056 -0.6774931922278133 -0.8820426996640419 -1.0864708470803943 -1.2908129231515748 -1.495102039757341 -1.6993691319825024 -1.903642958116921 -2.1079500996555116 -2.3123149612982417 -2.516759770950131 -2.7213045797212514 -2.9259672619267274 -3.1307635150867363 -3.335706859926509 -3.5408086403763264 -3.7460780235715228 -3.9515219998524853 -4.157145382764654 -4.362950809058522 -4.568938738689633 -4.775107454818583 -4.981453063811023 -5.187969495237655
do
	echo "changing current to $I"
	python change_I.py $I
	echo "running simulation"
	g4bl G4_M4_Mu2e_03.g4bl
	# copy data to new directory
	name1="sample_"
	# echo $partial
	# echo $I
	name=$name1$I
	echo $name
	mkdir $name
	echo "saving data..."
	for file in MW903.txt MW906.txt MW908.txt MW910.txt MW914.txt MW919.txt MW922.txt MW924.txt MW926.txt MW927.txt MW930.txt MW932.txt
	do
		# echo $file
		newname="$name/$file"
		echo "saving file $newname"
		cp $file $newname
	done
done
echo "Completed all runs!"


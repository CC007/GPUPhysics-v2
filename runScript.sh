#!/bin/bash
iterations=200

currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..9}
do
	currFolder=$(printf "%04d" $i)
	m ources/output/$currentDate/$currentTime/$currFolder
        dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder -f -gpu -i=$iterations -s < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

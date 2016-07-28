#!/bin/bash
iterations=1000

currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
echo running simulation p=1, no spin
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=1 -t=32 -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=3 t=32, no spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=3 -t=32 -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=3 t=8, no spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=3 -t=8 -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=7, no spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=7 -t=8 -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=1, with spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=1 -t=32 -s -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=3 t=32, with spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=3 -t=32 -s -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=3 t=8, with spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=3 -t=8 -s -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done

echo running simulation p=7, with spin
currentDate=`date +%Y-%m-%d`
currentTime=`date +%H%M%S`
for i in {0..1}
do
	currFolder=$(printf "%04d" $i)
	mkdir -p resources/output/$currentDate/$currentTime/$currFolder
	echo running simulation chunk $i
	nohup dist/simulation -m=resources/map/mapsy7.dat -c=resources/input/raw.input -o=resources/output/$currentDate/$currentTime/$currFolder/raw -f -i=$iterations -p=7 -t=8 -gpu < .enter 2>resources/output/$currentDate/$currentTime/$currFolder/log.output
done